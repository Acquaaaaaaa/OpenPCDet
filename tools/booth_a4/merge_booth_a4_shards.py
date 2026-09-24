"""Strictly validate and merge full-validation Booth A4 profile shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .build_frame_lists import read_frame_ids


METHODS = ("minmax", "p99_99", "p99_9")
SUM_COLUMNS = (
    "N_total", "N_A4", "N_exception", "N_zero_fp32_exact",
    "N_zero_quantized", "N_zero_from_rounding", "N_nonzero_quantized",
    "N_A4_nonzero_quantized", "N_clipped_low", "N_clipped_high", "call_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    expected_ids = read_frame_ids(args.manifest)
    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    shard_dirs = sorted(
        path for path in args.shards_dir.iterdir()
        if path.is_dir() and (path / "validation_checks.json").exists()
    )
    if not shard_dirs:
        raise ValueError("no complete shard directories found")

    all_frame_rows = []
    observed_ids = []
    reference_hashes = None
    registry_bytes = None
    int8_sum = {}
    digit_sum = {}
    for shard in shard_dirs:
        checks = json.loads((shard / "validation_checks.json").read_text(encoding="utf-8"))
        if checks.get("status") != "complete" or not checks.get("all_counter_invariants_passed"):
            raise ValueError(f"incomplete or invalid shard: {shard}")
        hashes = tuple(checks[key] for key in ("checkpoint_sha256", "config_sha256", "scales_sha256"))
        reference_hashes = hashes if reference_hashes is None else reference_hashes
        if hashes != reference_hashes:
            raise ValueError(f"configuration hash mismatch in shard: {shard}")
        current_registry = (shard / "activation_registry.json").read_bytes()
        registry_bytes = current_registry if registry_bytes is None else registry_bytes
        if current_registry != registry_bytes:
            raise ValueError(f"activation registry mismatch in shard: {shard}")
        observed_ids.extend(checks["processed_frame_ids"])
        all_frame_rows.extend(_read_csv(shard / "per_frame_counts.csv"))
        with np.load(shard / "int8_histograms.npz") as data:
            for key in data.files:
                int8_sum[key] = int8_sum.get(key, np.zeros(256, dtype=np.int64)) + data[key]
        with np.load(shard / "booth_digit_histograms.npz") as data:
            for key in data.files:
                digit_sum[key] = digit_sum.get(key, np.zeros((4, 5), dtype=np.int64)) + data[key]

    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("duplicate frame IDs across shards")
    if observed_ids != expected_ids:
        raise ValueError("shards do not exactly cover the full manifest in order")
    if reference_hashes[2] != sha256_file(args.scales):
        raise ValueError("provided scale file does not match shard scale hash")

    registry = json.loads(registry_bytes)
    edges = registry["activation_edges"]
    grouped = {}
    frame_ra4 = {}
    for row in all_frame_rows:
        key = (row["activation_id"], row["scale_method"])
        target = grouped.setdefault(key, {column: 0 for column in SUM_COLUMNS})
        for column in SUM_COLUMNS:
            target[column] += int(row[column])
        frame_ra4.setdefault(key, []).append(int(row["N_A4"]) / int(row["N_total"]))

    summary_rows = []
    for order, edge in enumerate(edges):
        activation_id = edge["activation_id"]
        for method in METHODS:
            key = (activation_id, method)
            if key not in grouped:
                continue
            counts = grouped[key]
            hist_key = f"{activation_id}__{method}"
            int8_hist = int8_sum[hist_key]
            digit_hist = digit_sum[hist_key]
            n_total = counts["N_total"]
            if counts["N_A4"] + counts["N_exception"] != n_total:
                raise ValueError(f"A4 count invariant failed for {key}")
            if int8_hist.sum() != n_total or not np.all(digit_hist.sum(axis=1) == n_total):
                raise ValueError(f"histogram invariant failed for {key}")
            nonzero_bins = np.flatnonzero(int8_hist)
            values = np.asarray(frame_ra4[key], dtype=np.float64)
            summary_rows.append({
                "activation_order": order,
                "activation_id": activation_id,
                "producer": edge["producer"],
                "consumer_layers": json.dumps(edge["consumer_layers"]),
                "scale_method": method,
                "threshold": scales[activation_id][method]["threshold"],
                "scale": scales[activation_id][method]["scale"],
                **counts,
                "RA4_tensor_micro": counts["N_A4"] / n_total,
                "Rexc_tensor_micro": counts["N_exception"] / n_total,
                "Rzero_fp32": counts["N_zero_fp32_exact"] / n_total,
                "Rzero_quantized": counts["N_zero_quantized"] / n_total,
                "Rzero_from_rounding": counts["N_zero_from_rounding"] / n_total,
                "RA4_nonzero_q": (
                    counts["N_A4_nonzero_quantized"] / counts["N_nonzero_quantized"]
                    if counts["N_nonzero_quantized"] else None
                ),
                "Rclip": (counts["N_clipped_low"] + counts["N_clipped_high"]) / n_total,
                "N_q_minus128": int(int8_hist[0]),
                "q_min": int(nonzero_bins[0] - 128),
                "q_max": int(nonzero_bins[-1] - 128),
                "d2_zero_ratio": float(digit_hist[2, 2] / n_total),
                "d3_zero_ratio": float(digit_hist[3, 2] / n_total),
                "macro_mean": float(values.mean()),
                "macro_p5": float(np.percentile(values, 5)),
                "macro_p50": float(np.percentile(values, 50)),
                "macro_p95": float(np.percentile(values, 95)),
                "macro_min": float(values.min()),
                "macro_max": float(values.max()),
            })

    _write_csv(args.output_dir / "activation_summary_long.csv", summary_rows)
    _write_csv(args.output_dir / "per_frame_counts.csv", all_frame_rows)
    np.savez_compressed(args.output_dir / "int8_histograms.npz", **int8_sum)
    np.savez_compressed(args.output_dir / "booth_digit_histograms.npz", **digit_sum)
    for filename in ("activation_registry.json", "activation_edges.csv", "layer_consumers.csv"):
        shutil.copy2(shard_dirs[0] / filename, args.output_dir / filename)
    checks = {
        "status": "complete",
        "shard_count": len(shard_dirs),
        "processed_frame_count": len(observed_ids),
        "unique_frame_count": len(set(observed_ids)),
        "frame_order_exact": observed_ids == expected_ids,
        "all_counter_invariants_passed": True,
        "manifest_sha256": sha256_file(args.manifest),
        "scales_sha256": sha256_file(args.scales),
        "checkpoint_sha256": reference_hashes[0],
        "config_sha256": reference_hashes[1],
    }
    (args.output_dir / "validation_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
