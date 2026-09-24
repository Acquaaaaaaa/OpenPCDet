"""Profile PointPillars Booth groups and A/B/C cycles on reference activations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.detectors.detector3d_template import _load_checkpoint
from pcdet.utils import common_utils
from tools.booth_a4.activation_registry import build_activation_registry
from tools.booth_a4.booth_radix4 import encode_modified_radix4
from tools.booth_a4.build_frame_lists import read_frame_ids
from tools.booth_a4.cycle_simulator import CycleConfig, simulate_cycles
from tools.booth_a4.group_counter import GroupCounts, count_group_batches
from tools.booth_a4.operand_mapper import iter_consumer_groups, output_channels
from tools.booth_a4.quantizer import quantize_symmetric_int8
from tools.booth_a4.report_group_cycle import generate_reports, write_aggregate_records


DEFAULT_SCALE_METHODS = ("minmax", "p99_99", "p99_9")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_id(info: dict) -> str:
    value = info.get("point_cloud", {}).get("lidar_idx")
    if value is None:
        value = info.get("image", {}).get("image_idx")
    if value is None:
        raise KeyError("KITTI info lacks a frame ID")
    return str(value).zfill(6)


def _filter_dataset(dataset, requested_ids: list[str]) -> None:
    by_id = {_frame_id(info): info for info in dataset.kitti_infos}
    missing = [frame_id for frame_id in requested_ids if frame_id not in by_id]
    if missing:
        raise ValueError(f"manifest IDs missing from dataset infos: {missing[:10]}")
    dataset.kitti_infos = [by_id[frame_id] for frame_id in requested_ids]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _valid_point_mask(activation: torch.Tensor, voxel_num_points: torch.Tensor) -> torch.Tensor:
    if activation.ndim != 3:
        raise ValueError("PFN activation must have shape [voxels, point_slots, features]")
    if voxel_num_points.ndim != 1 or voxel_num_points.shape[0] != activation.shape[0]:
        raise ValueError("voxel_num_points does not match the PFN activation")
    slots = torch.arange(activation.shape[1], device=activation.device)
    return slots[None, :] < voxel_num_points.to(device=activation.device)[:, None]


class GroupCycleProfiler:
    def __init__(
        self,
        model: nn.Module,
        registry,
        scales: dict,
        *,
        scale_methods: tuple[str, ...],
        group_size: int,
        chunk_streams: int,
        mapping_mode: str,
        conv_transpose_mapping: str,
        padding_policy: str,
        pfn_token_policy: str,
        window_sizes: tuple[int, ...],
        metrics_mode: str,
        primary_cycle_config: CycleConfig,
    ) -> None:
        self.model = model
        self.registry = registry
        self.scales = scales
        self.scale_methods = scale_methods
        self.group_size = group_size
        self.chunk_streams = chunk_streams
        self.mapping_mode = mapping_mode
        self.conv_transpose_mapping = conv_transpose_mapping
        self.padding_policy = padding_policy
        self.pfn_token_policy = pfn_token_policy
        self.window_sizes = window_sizes
        self.metrics_mode = metrics_mode
        self.primary_cycle_config = primary_cycle_config
        self.module_map = dict(model.named_modules())
        self.consumer_map = {record.consumer_layer: record for record in registry.consumers}
        self.edge_map = {edge.activation_id: edge for edge in registry.edges}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.current_frame_id: str | None = None
        self.current_voxel_num_points: torch.Tensor | None = None
        self.frame_rows: list[dict[str, Any]] = []
        self.frame_seen: set[tuple[str, str]] = set()
        self.aggregate: dict[tuple[str, str], GroupCounts] = {}
        self.skipped_methods: list[dict[str, str]] = []
        self.valid_methods: dict[str, tuple[str, ...]] = {}
        for edge in registry.edges:
            available = []
            if edge.activation_id not in scales:
                raise KeyError(f"scale file is missing activation: {edge.activation_id}")
            for method in scale_methods:
                method_data = scales[edge.activation_id].get(method)
                if method_data and method_data.get("status") == "ok" and method_data.get("scale"):
                    available.append(method)
                else:
                    self.skipped_methods.append({
                        "activation_id": edge.activation_id,
                        "scale_method": method,
                        "status": "unavailable",
                    })
            self.valid_methods[edge.activation_id] = tuple(available)

    def start_frame(self, frame_id: str, batch_dict: dict) -> None:
        self.current_frame_id = frame_id
        self.current_voxel_num_points = batch_dict.get("voxel_num_points")
        self.frame_rows = []
        self.frame_seen = set()

    def _effective_padding(self, edge, module: nn.Module):
        if not isinstance(module, nn.Conv2d) or edge.capture_strategy != "pre_zero_pad_input":
            return None
        pad_module = self.module_map[edge.capture_module]
        if not isinstance(pad_module, nn.ZeroPad2d):
            raise TypeError("pre_zero_pad_input edge does not reference nn.ZeroPad2d")
        return pad_module.padding

    def _metadata(self, consumer, method: str, module: nn.Module) -> dict[str, Any]:
        return {
            "layer_order": consumer.layer_order,
            "consumer_layer": consumer.consumer_layer,
            "activation_id": consumer.activation_id,
            "module_type": consumer.module_type,
            "weight_shape": json.dumps(consumer.weight_shape),
            "scale_method": method,
            "mapping_mode": self.mapping_mode,
            "conv_transpose_mapping": self.conv_transpose_mapping,
            "padding_policy": self.padding_policy,
            "pfn_token_policy": self.pfn_token_policy,
            "N_main": self.group_size,
            "C_out": output_channels(module),
        }

    def _profile_consumer(
        self,
        digits: torch.Tensor,
        edge,
        consumer,
        method: str,
        valid_token_mask: torch.Tensor | None,
    ) -> None:
        module = self.module_map[consumer.consumer_layer]
        batches = iter_consumer_groups(
            digits,
            module,
            group_size=self.group_size,
            chunk_streams=self.chunk_streams,
            padding_policy=self.padding_policy,
            token_policy=self.pfn_token_policy,
            valid_token_mask=valid_token_mask if isinstance(module, nn.Linear) else None,
            effective_padding=self._effective_padding(edge, module),
            mapping_mode=self.mapping_mode,
            conv_transpose_mapping=self.conv_transpose_mapping,
        )
        counts = count_group_batches(
            batches,
            group_size=self.group_size,
            window_sizes=self.window_sizes,
            retain_sensitivity=self.metrics_mode == "full",
        )
        key = (consumer.consumer_layer, method)
        if key in self.frame_seen:
            raise RuntimeError(
                f"consumer observed more than once in frame {self.current_frame_id}: {key}"
            )
        self.frame_seen.add(key)
        aggregate = self.aggregate.setdefault(
            key,
            GroupCounts(
                group_size=self.group_size,
                window_sizes=self.window_sizes if self.metrics_mode == "full" else (),
                sensitivity_retained=self.metrics_mode == "full",
            ),
        )
        aggregate.add(counts)
        metadata = self._metadata(consumer, method, module)
        cycle_row = simulate_cycles(
            counts,
            output_channels=output_channels(module),
            config=self.primary_cycle_config,
        )
        self.frame_rows.append({
            "frame_id": self.current_frame_id,
            **metadata,
            **counts.metrics(),
            **cycle_row,
        })

    def _hook(self, edge):
        def profile(_module: nn.Module, inputs: tuple) -> None:
            if self.current_frame_id is None:
                raise RuntimeError("start_frame must be called before model execution")
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"{edge.activation_id} did not receive tensor inputs[0]")
            activation = inputs[0].detach()
            valid_token_mask = None
            if any(
                isinstance(self.module_map[name], nn.Linear)
                for name in edge.consumer_layers
            ):
                if self.current_voxel_num_points is None:
                    raise KeyError("batch_dict lacks voxel_num_points for PFN token masking")
                valid_token_mask = _valid_point_mask(
                    activation, self.current_voxel_num_points
                )
            for method in self.valid_methods[edge.activation_id]:
                scale = self.scales[edge.activation_id][method]["scale"]
                quantized = quantize_symmetric_int8(activation, scale)
                digits = encode_modified_radix4(quantized.q)
                for consumer_name in edge.consumer_layers:
                    self._profile_consumer(
                        digits,
                        edge,
                        self.consumer_map[consumer_name],
                        method,
                        valid_token_mask,
                    )
        return profile

    def __enter__(self) -> "GroupCycleProfiler":
        for edge in self.registry.edges:
            self.handles.append(
                self.module_map[edge.capture_module].register_forward_pre_hook(self._hook(edge))
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finish_frame_rows(self) -> list[dict[str, Any]]:
        expected = {
            (consumer.consumer_layer, method)
            for consumer in self.registry.consumers
            for method in self.valid_methods[consumer.activation_id]
        }
        missing = sorted(expected - self.frame_seen)
        if missing:
            raise ValueError(
                f"frame {self.current_frame_id} missed consumer-scale records: {missing[:10]}"
            )
        return sorted(
            self.frame_rows,
            key=lambda row: (row["layer_order"], row["scale_method"]),
        )

    def aggregate_records(self) -> list[dict[str, Any]]:
        records = []
        for consumer in self.registry.consumers:
            module = self.module_map[consumer.consumer_layer]
            for method in self.valid_methods[consumer.activation_id]:
                counts = self.aggregate[(consumer.consumer_layer, method)]
                records.append({
                    **self._metadata(consumer, method, module),
                    "counts": counts.to_dict(),
                })
        return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--correction-parallelism", type=int, default=8)
    parser.add_argument("--scalar-correction-parallelism", type=int, default=8)
    parser.add_argument("--main-output-parallelism", type=int)
    parser.add_argument("--correction-output-parallelism", type=int)
    parser.add_argument("--padding-policy", choices=(
        "physical_rows_included", "ideal_padding_gated"
    ), default="physical_rows_included")
    parser.add_argument("--pfn-token-policy", choices=(
        "fixed_slots_included", "valid_points_only"
    ), default="fixed_slots_included")
    parser.add_argument("--chunk-streams", type=int, default=16384)
    parser.add_argument("--mapping-mode", choices=(
        "consumer_aware_reference_cim", "logical_contiguous"
    ), default="consumer_aware_reference_cim")
    parser.add_argument("--conv-transpose-mapping", choices=(
        "direct_scatter", "zero_insertion"
    ), default="direct_scatter")
    parser.add_argument("--window-sizes", type=int, nargs="+", default=(8, 32, 128))
    parser.add_argument("--scale-methods", nargs="+", default=DEFAULT_SCALE_METHODS)
    parser.add_argument("--metrics-mode", choices=("full", "core"), default="full")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("cfg_file", "checkpoint", "manifest", "scales", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PointPillars profiling")
    if args.group_size <= 0 or args.correction_parallelism <= 0 or args.chunk_streams <= 0:
        raise ValueError("group size, correction parallelism, and chunk size must be positive")

    all_manifest_ids = read_frame_ids(args.manifest)
    if args.start_index < 0 or args.start_index >= len(all_manifest_ids):
        raise ValueError("--start-index is outside the manifest")
    requested_ids = all_manifest_ids[args.start_index :]
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        requested_ids = requested_ids[: args.max_frames]
    scales = json.loads(args.scales.read_text(encoding="utf-8"))

    os.chdir(REPO_ROOT / "tools")
    cfg_from_yaml_file(str(args.cfg_file), cfg)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logger = common_utils.create_logger(args.output_dir / "profile.log", rank=0)
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    _filter_dataset(dataset, requested_ids)
    model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    checkpoint = _load_checkpoint(str(args.checkpoint), map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.cuda().eval()
    registry = build_activation_registry(model)
    registry.write(args.output_dir)

    primary_cycle_config = CycleConfig(
        correction_parallelism=args.correction_parallelism,
        main_output_parallelism=args.main_output_parallelism,
        correction_output_parallelism=args.correction_output_parallelism,
        scalar_correction_parallelism=args.scalar_correction_parallelism,
    )
    profiler = GroupCycleProfiler(
        model,
        registry,
        scales,
        scale_methods=tuple(args.scale_methods),
        group_size=args.group_size,
        chunk_streams=args.chunk_streams,
        mapping_mode=args.mapping_mode,
        conv_transpose_mapping=args.conv_transpose_mapping,
        padding_policy=args.padding_policy,
        pfn_token_policy=args.pfn_token_policy,
        window_sizes=tuple(args.window_sizes),
        metrics_mode=args.metrics_mode,
        primary_cycle_config=primary_cycle_config,
    )
    per_frame_rows: list[dict[str, Any]] = []
    processed_ids: list[str] = []
    started = time.perf_counter()
    with profiler, torch.inference_mode():
        for index, batch in enumerate(loader):
            frame_id = str(batch["frame_id"][0])
            if frame_id != requested_ids[index]:
                raise AssertionError(f"frame order mismatch: {frame_id} != {requested_ids[index]}")
            load_data_to_gpu(batch)
            profiler.start_frame(frame_id, batch)
            for module in model.module_list:
                batch = module(batch)
            per_frame_rows.extend(profiler.finish_frame_rows())
            processed_ids.append(frame_id)
            if (index + 1) % 16 == 0 or index + 1 == len(requested_ids):
                print(f"group-cycle {index + 1}/{len(requested_ids)} frame={frame_id}", flush=True)
    elapsed = time.perf_counter() - started

    aggregate_records = profiler.aggregate_records()
    write_aggregate_records(args.output_dir / "aggregate_group_counts.json", aggregate_records)
    _write_csv(args.output_dir / "per_frame_consumer_counts.csv", per_frame_rows)
    report_summary = generate_reports(
        aggregate_records,
        args.output_dir,
        main_output_parallelism=args.main_output_parallelism,
        correction_output_parallelism=args.correction_output_parallelism,
        scalar_correction_parallelism=args.scalar_correction_parallelism,
        include_sensitivity=args.metrics_mode == "full",
    )
    resolved_config = {
        "group_size": args.group_size,
        "correction_parallelism": args.correction_parallelism,
        "scalar_correction_parallelism": args.scalar_correction_parallelism,
        "main_output_parallelism": args.main_output_parallelism,
        "correction_output_parallelism": args.correction_output_parallelism,
        "padding_policy": args.padding_policy,
        "pfn_token_policy": args.pfn_token_policy,
        "chunk_streams": args.chunk_streams,
        "mapping_mode": args.mapping_mode,
        "conv_transpose_mapping": args.conv_transpose_mapping,
        "window_sizes": args.window_sizes if args.metrics_mode == "full" else [],
        "window_sizes_requested": args.window_sizes,
        "scale_methods": args.scale_methods,
        "metrics_mode": args.metrics_mode,
        "sensitivity_retained": args.metrics_mode == "full",
        "correction_vector_item_semantics": (
            "one activation digit lane applies to one complete correction output tile"
        ),
        "scalar_item_semantics": "one activation digit times one output-channel weight",
        "output_tile_cross_packing": False,
    }
    resolved_config_path = args.output_dir / "resolved_group_cycle_config.json"
    resolved_config_path.write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checks = {
        "status": "complete",
        "formal_full_manifest": args.start_index == 0 and args.max_frames is None,
        "source_manifest_frame_count": len(all_manifest_ids),
        "requested_frame_count": len(requested_ids),
        "processed_frame_count": len(processed_ids),
        "processed_frame_ids": processed_ids,
        "frame_order_exact": processed_ids == requested_ids,
        "consumer_count": len(registry.consumers),
        "activation_count": len(registry.edges),
        "aggregate_record_count": len(aggregate_records),
        "per_frame_record_count": len(per_frame_rows),
        "all_group_counter_invariants_passed": True,
        "report_summary": report_summary,
        "skipped_scale_methods": profiler.skipped_methods,
        "metrics_mode": args.metrics_mode,
        "manifest_sha256": sha256_file(args.manifest),
        "scales_sha256": sha256_file(args.scales),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.cfg_file),
        "group_cycle_config_sha256": sha256_file(resolved_config_path),
        "elapsed_seconds": elapsed,
    }
    (args.output_dir / "validation_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "processed_frame_count": len(processed_ids),
        "aggregate_record_count": len(aggregate_records),
        "elapsed_seconds": elapsed,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
