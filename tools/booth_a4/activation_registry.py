"""PointPillars MAC-consumer and logical-activation registry construction."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import torch
import torch.nn as nn


_BLOCK_CONV = re.compile(r"^backbone_2d\.blocks\.(\d+)\.(\d+)$")
_DEBLOCK_CONV = re.compile(r"^backbone_2d\.deblocks\.(\d+)\.(\d+)$")
_DENSE_HEADS = {"dense_head.conv_cls", "dense_head.conv_box", "dense_head.conv_dir_cls"}


def _activation_id(label: str) -> str:
    return "act__" + re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")


@dataclass(frozen=True)
class ConsumerRecord:
    layer_order: int
    consumer_layer: str
    module_type: str
    weight_shape: tuple[int, ...]
    activation_id: str
    hook_strategy: str
    shared_activation: bool


@dataclass(frozen=True)
class ActivationEdgeRecord:
    activation_id: str
    producer: str
    transform_chain: str
    consumer_layers: tuple[str, ...]
    shared_consumer_count: int
    exclude_boundary_padding: bool
    capture_module: str
    capture_strategy: str


@dataclass(frozen=True)
class ActivationRegistry:
    consumers: tuple[ConsumerRecord, ...]
    edges: tuple[ActivationEdgeRecord, ...]

    def to_dict(self) -> dict:
        return {
            "consumer_count": len(self.consumers),
            "unique_activation_count": len(self.edges),
            "consumers": [asdict(record) for record in self.consumers],
            "activation_edges": [asdict(record) for record in self.edges],
        }

    def write(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "activation_registry.json").write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_csv(destination / "layer_consumers.csv", self.consumers)
        _write_csv(destination / "activation_edges.csv", self.edges)


class ActivationShapeObserver:
    """Non-mutating pre-hook observer for one sample-forward registry check."""

    def __init__(self, model: nn.Module, registry: ActivationRegistry) -> None:
        self.model = model
        self.registry = registry
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.observations = {
            edge.activation_id: {
                "call_count": 0,
                "empty_call_count": 0,
                "shape_patterns": [],
                "dtypes": [],
                "devices": [],
            }
            for edge in registry.edges
        }

    def _hook(self, activation_id: str):
        def observe(_module: nn.Module, inputs: tuple) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"{activation_id} hook did not receive a tensor as inputs[0]")
            tensor = inputs[0]
            record = self.observations[activation_id]
            record["call_count"] += 1
            record["empty_call_count"] += int(tensor.numel() == 0)
            for key, value in (
                ("shape_patterns", list(tensor.shape)),
                ("dtypes", str(tensor.dtype)),
                ("devices", str(tensor.device)),
            ):
                if value not in record[key]:
                    record[key].append(value)

        return observe

    def __enter__(self) -> "ActivationShapeObserver":
        module_map = dict(self.model.named_modules())
        capture_modules: set[str] = set()
        for edge in self.registry.edges:
            if edge.capture_module in capture_modules:
                raise ValueError(f"duplicate capture module in registry: {edge.capture_module}")
            capture_modules.add(edge.capture_module)
            self.handles.append(
                module_map[edge.capture_module].register_forward_pre_hook(
                    self._hook(edge.activation_id)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def validate(self) -> None:
        missing = [
            activation_id
            for activation_id, record in self.observations.items()
            if record["call_count"] == 0
        ]
        if missing:
            raise ValueError(f"registered activations were not observed: {missing}")


def _write_csv(path: Path, records: Iterable[object]) -> None:
    rows = [asdict(record) for record in records]
    if not rows:
        raise ValueError(f"refusing to write empty registry table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (tuple, list)) else value
                    for key, value in row.items()
                }
            )


def _is_target_consumer(name: str, module: nn.Module) -> bool:
    if name.startswith("vfe.") and isinstance(module, nn.Linear):
        return True
    if name.startswith("backbone_2d.") and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return True
    return name in _DENSE_HEADS and isinstance(module, nn.Conv2d)


def _first_conv_indices(model: nn.Module) -> dict[int, int]:
    first: dict[int, int] = {}
    for name, module in model.named_modules():
        match = _BLOCK_CONV.match(name)
        if match and isinstance(module, nn.Conv2d):
            block_index, child_index = map(int, match.groups())
            first[block_index] = min(first.get(block_index, child_index), child_index)
    return first


def _preceding_zero_pad(
    module_map: dict[str, nn.Module],
    consumer_layer: str,
) -> str | None:
    parent_path, child = consumer_layer.rsplit(".", 1)
    if not child.isdigit() or int(child) == 0:
        return None
    candidate = f"{parent_path}.{int(child) - 1}"
    return candidate if isinstance(module_map.get(candidate), nn.ZeroPad2d) else None


def _classify_edge(
    consumer_layer: str,
    module_map: dict[str, nn.Module],
    first_conv: dict[int, int],
) -> tuple[str, str, str, bool, str, str]:
    """Return id, producer, transform, excludes-pad, capture-module, strategy."""

    if consumer_layer.startswith("vfe."):
        return (
            _activation_id(consumer_layer + "_input"),
            "vfe_feature_construction",
            "feature_concat -> padding_mask",
            False,
            consumer_layer,
            "consumer_pre_hook",
        )

    if consumer_layer in _DENSE_HEADS:
        return (
            _activation_id("dense_head_spatial_features_2d"),
            "backbone_2d",
            "deblock outputs -> concat",
            False,
            "dense_head.conv_cls",
            "shared_owner_pre_hook",
        )

    block_match = _BLOCK_CONV.match(consumer_layer)
    if block_match:
        block_index, child_index = map(int, block_match.groups())
        if child_index == first_conv[block_index]:
            pad_module = _preceding_zero_pad(module_map, consumer_layer)
            if pad_module is None:
                raise ValueError(
                    f"first stage Conv2d {consumer_layer} is not preceded by nn.ZeroPad2d; "
                    "the logical padding-exclusion rule needs an explicit implementation"
                )
            if block_index == 0:
                activation_id = _activation_id("backbone_2d_spatial_features")
                producer = "map_to_bev_module"
            else:
                activation_id = _activation_id(f"backbone_2d_stage_{block_index - 1}_output")
                producer = f"backbone_2d.blocks.{block_index - 1}"
            return (
                activation_id,
                producer,
                "logical tensor before ZeroPad2d",
                True,
                pad_module,
                "pre_zero_pad_input",
            )
        parent, child = consumer_layer.rsplit(".", 1)
        previous = f"{parent}.{int(child) - 1}"
        return (
            _activation_id(consumer_layer + "_input"),
            previous,
            "previous Conv2d -> BatchNorm2d -> ReLU",
            True,
            consumer_layer,
            "consumer_pre_hook",
        )

    deblock_match = _DEBLOCK_CONV.match(consumer_layer)
    if deblock_match:
        stage_index, _ = map(int, deblock_match.groups())
        return (
            _activation_id(f"backbone_2d_stage_{stage_index}_output"),
            f"backbone_2d.blocks.{stage_index}",
            "stage output shared with deblock and next stage when present",
            False,
            consumer_layer,
            "consumer_pre_hook",
        )

    raise ValueError(f"unclassified target consumer: {consumer_layer}")


def build_activation_registry(model: nn.Module) -> ActivationRegistry:
    """Build a deterministic logical-edge registry for PointPillars."""

    module_map = dict(model.named_modules())
    first_conv = _first_conv_indices(model)
    target_modules = [
        (name, module)
        for name, module in model.named_modules()
        if _is_target_consumer(name, module)
    ]
    if not target_modules:
        raise ValueError("no PointPillars MAC consumers were found")

    pending_consumers: list[dict] = []
    pending_edges: dict[str, dict] = {}
    for order, (name, module) in enumerate(target_modules):
        (
            activation_id,
            producer,
            transform_chain,
            exclude_boundary_padding,
            capture_module,
            capture_strategy,
        ) = _classify_edge(name, module_map, first_conv)
        if capture_module not in module_map:
            raise ValueError(f"capture module does not exist: {capture_module}")

        edge = pending_edges.setdefault(
            activation_id,
            {
                "activation_id": activation_id,
                "producer": producer,
                "transform_chain": transform_chain,
                "consumer_layers": [],
                "exclude_boundary_padding": exclude_boundary_padding,
                "capture_module": capture_module,
                "capture_strategy": capture_strategy,
            },
        )
        edge["consumer_layers"].append(name)
        edge["exclude_boundary_padding"] = (
            edge["exclude_boundary_padding"] or exclude_boundary_padding
        )
        pending_consumers.append(
            {
                "layer_order": order,
                "consumer_layer": name,
                "module_type": type(module).__name__,
                "weight_shape": tuple(module.weight.shape),
                "activation_id": activation_id,
                "hook_strategy": capture_strategy,
            }
        )

    shared = {
        activation_id: len(edge["consumer_layers"]) > 1
        for activation_id, edge in pending_edges.items()
    }
    consumers = tuple(
        ConsumerRecord(**record, shared_activation=shared[record["activation_id"]])
        for record in pending_consumers
    )
    edges = tuple(
        ActivationEdgeRecord(
            activation_id=edge["activation_id"],
            producer=edge["producer"],
            transform_chain=edge["transform_chain"],
            consumer_layers=tuple(edge["consumer_layers"]),
            shared_consumer_count=len(edge["consumer_layers"]),
            exclude_boundary_padding=edge["exclude_boundary_padding"],
            capture_module=edge["capture_module"],
            capture_strategy=edge["capture_strategy"],
        )
        for edge in pending_edges.values()
    )
    return ActivationRegistry(consumers=consumers, edges=edges)
