"""Profile frozen-scale INT8 Booth A4 coverage on PointPillars activations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.detectors.detector3d_template import _load_checkpoint
from pcdet.utils import common_utils
from tools.booth_a4.activation_registry import build_activation_registry
from tools.booth_a4.build_frame_lists import read_frame_ids
from tools.booth_a4.counter import BoothA4Accumulator, count_activation_chunked


SCALE_METHODS = ("minmax", "p99_99", "p99_9")


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


class ActivationBoothProfiler:
    def __init__(self, model, registry, scales, *, chunk_elements):
        self.model = model
        self.registry = registry
        self.scales = scales
        self.chunk_elements = chunk_elements
        self.current_frame_id: str | None = None
        self.global_counts = {}
        self.frame_counts = {}
        self.handles = []
        self.skipped_methods = []
        for edge in registry.edges:
            activation_id = edge.activation_id
            if activation_id not in scales:
                raise KeyError(f"scale file is missing activation: {activation_id}")
            for method in SCALE_METHODS:
                method_data = scales[activation_id][method]
                if method_data["status"] == "ok" and method_data["scale"] is not None:
                    self.global_counts[(activation_id, method)] = BoothA4Accumulator()
                else:
                    self.skipped_methods.append(
                        {
                            "activation_id": activation_id,
                            "scale_method": method,
                            "status": method_data["status"],
                        }
                    )

    def start_frame(self, frame_id: str) -> None:
        self.current_frame_id = frame_id
        self.frame_counts = {
            key: BoothA4Accumulator() for key in self.global_counts
        }

    def _hook(self, activation_id):
        def profile(_module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"{activation_id} did not receive tensor inputs[0]")
            activation = inputs[0]
            for method in SCALE_METHODS:
                key = (activation_id, method)
                if key not in self.global_counts:
                    continue
                scale = self.scales[activation_id][method]["scale"]
                try:
                    counts = count_activation_chunked(
                        activation,
                        scale,
                        chunk_elements=self.chunk_elements,
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"profiling failed at frame={self.current_frame_id}, "
                        f"activation_id={activation_id}, scale_method={method}: {error}"
                    ) from error
                self.global_counts[key].add(counts)
                self.frame_counts[key].add(counts)

        return profile

    def __enter__(self):
        module_map = dict(self.model.named_modules())
        for edge in self.registry.edges:
            self.handles.append(
                module_map[edge.capture_module].register_forward_pre_hook(
                    self._hook(edge.activation_id)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finish_frame_rows(self) -> list[dict]:
        rows = []
        for (activation_id, method), accumulator in self.frame_counts.items():
            counts = accumulator.counts
            counts.validate()
            if counts.call_count == 0:
                raise ValueError(
                    f"activation not observed in frame {self.current_frame_id}: {activation_id}"
                )
            rows.append(
                {
                    "frame_id": self.current_frame_id,
                    "activation_id": activation_id,
                    "scale_method": method,
                    "N_total": counts.n_total,
                    "N_A4": counts.n_a4,
                    "N_exception": counts.n_exception,
                    "N_zero_fp32_exact": counts.n_zero_fp32_exact,
                    "N_zero_quantized": counts.n_zero_quantized,
                    "N_zero_from_rounding": counts.n_zero_from_rounding,
                    "N_nonzero_quantized": counts.n_nonzero_quantized,
                    "N_A4_nonzero_quantized": counts.n_a4_nonzero_quantized,
                    "N_clipped_low": counts.n_clipped_low,
                    "N_clipped_high": counts.n_clipped_high,
                    "call_count": counts.call_count,
                    "RA4_frame": counts.n_a4 / counts.n_total,
                }
            )
        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-elements", type=int, default=4_000_000)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(profiler, registry, scales, per_frame_rows):
    edge_map = {edge.activation_id: edge for edge in registry.edges}
    grouped_frame_ra4 = {}
    for row in per_frame_rows:
        grouped_frame_ra4.setdefault(
            (row["activation_id"], row["scale_method"]), []
        ).append(row["RA4_frame"])

    rows = []
    for (activation_id, method), accumulator in profiler.global_counts.items():
        counts = accumulator.counts
        counts.validate()
        edge = edge_map[activation_id]
        frame_values = np.asarray(grouped_frame_ra4[(activation_id, method)], dtype=np.float64)
        row = {
            "activation_order": list(edge_map).index(activation_id),
            "activation_id": activation_id,
            "producer": edge.producer,
            "consumer_layers": json.dumps(edge.consumer_layers),
            "scale_method": method,
            "threshold": scales[activation_id][method]["threshold"],
            "scale": scales[activation_id][method]["scale"],
            "N_total": counts.n_total,
            "N_A4": counts.n_a4,
            "N_exception": counts.n_exception,
            "RA4_tensor_micro": counts.n_a4 / counts.n_total,
            "Rexc_tensor_micro": counts.n_exception / counts.n_total,
            "N_zero_fp32_exact": counts.n_zero_fp32_exact,
            "N_zero_quantized": counts.n_zero_quantized,
            "N_zero_from_rounding": counts.n_zero_from_rounding,
            "N_nonzero_quantized": counts.n_nonzero_quantized,
            "N_A4_nonzero_quantized": counts.n_a4_nonzero_quantized,
            "Rzero_fp32": counts.n_zero_fp32_exact / counts.n_total,
            "Rzero_quantized": counts.n_zero_quantized / counts.n_total,
            "Rzero_from_rounding": counts.n_zero_from_rounding / counts.n_total,
            "RA4_nonzero_q": (
                counts.n_a4_nonzero_quantized / counts.n_nonzero_quantized
                if counts.n_nonzero_quantized else None
            ),
            "N_clipped_low": counts.n_clipped_low,
            "N_clipped_high": counts.n_clipped_high,
            "Rclip": (counts.n_clipped_low + counts.n_clipped_high) / counts.n_total,
            "N_q_minus128": counts.n_q_minus128,
            "q_min": counts.q_min,
            "q_max": counts.q_max,
            "d2_zero_ratio": counts.digit_hist[2][2] / counts.n_total,
            "d3_zero_ratio": counts.digit_hist[3][2] / counts.n_total,
            "call_count": counts.call_count,
            "macro_mean": float(frame_values.mean()),
            "macro_p5": float(np.percentile(frame_values, 5)),
            "macro_p50": float(np.percentile(frame_values, 50)),
            "macro_p95": float(np.percentile(frame_values, 95)),
            "macro_min": float(frame_values.min()),
            "macro_max": float(frame_values.max()),
        }
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    for name in ("cfg_file", "checkpoint", "manifest", "scales", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PointPillars profiling")

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

    profiler = ActivationBoothProfiler(
        model, registry, scales, chunk_elements=args.chunk_elements
    )
    per_frame_rows = []
    processed_ids = []
    started = time.perf_counter()
    with profiler, torch.inference_mode():
        for index, batch in enumerate(loader):
            frame_id = str(batch["frame_id"][0])
            if frame_id != requested_ids[index]:
                raise AssertionError(f"frame order mismatch: {frame_id} != {requested_ids[index]}")
            profiler.start_frame(frame_id)
            load_data_to_gpu(batch)
            for module in model.module_list:
                batch = module(batch)
            per_frame_rows.extend(profiler.finish_frame_rows())
            processed_ids.append(frame_id)
            if (index + 1) % 16 == 0 or index + 1 == len(requested_ids):
                print(f"profile {index + 1}/{len(requested_ids)} frame={frame_id}", flush=True)
    elapsed = time.perf_counter() - started

    summary_rows = _summary_rows(profiler, registry, scales, per_frame_rows)
    _write_csv(args.output_dir / "activation_summary_long.csv", summary_rows)
    _write_csv(args.output_dir / "per_frame_counts.csv", per_frame_rows)
    int8_histograms = {
        f"{activation_id}__{method}": np.asarray(accumulator.counts.int8_hist, dtype=np.int64)
        for (activation_id, method), accumulator in profiler.global_counts.items()
    }
    digit_histograms = {
        f"{activation_id}__{method}": np.asarray(accumulator.counts.digit_hist, dtype=np.int64)
        for (activation_id, method), accumulator in profiler.global_counts.items()
    }
    np.savez_compressed(args.output_dir / "int8_histograms.npz", **int8_histograms)
    np.savez_compressed(args.output_dir / "booth_digit_histograms.npz", **digit_histograms)

    checks = {
        "status": "complete",
        "formal_full_manifest": args.start_index == 0 and args.max_frames is None,
        "source_manifest_frame_count": len(all_manifest_ids),
        "start_index": args.start_index,
        "end_index_inclusive": args.start_index + len(processed_ids) - 1,
        "requested_frame_count": len(requested_ids),
        "processed_frame_count": len(processed_ids),
        "processed_frame_ids": processed_ids,
        "frame_order_exact": processed_ids == requested_ids,
        "activation_count": len(registry.edges),
        "scale_result_count": len(summary_rows),
        "skipped_scale_methods": profiler.skipped_methods,
        "all_counter_invariants_passed": True,
        "booth_oracle_conflicts": 0,
        "manifest_sha256": sha256_file(args.manifest),
        "scales_sha256": sha256_file(args.scales),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.cfg_file),
        "elapsed_seconds": elapsed,
        "batch_size": 1,
        "workers": args.workers,
        "seed": args.seed,
        "chunk_elements": args.chunk_elements,
    }
    (args.output_dir / "validation_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "processed_frame_count": len(processed_ids),
        "result_rows": len(summary_rows),
        "elapsed_seconds": elapsed,
    }))


if __name__ == "__main__":
    main()
