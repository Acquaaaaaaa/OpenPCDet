"""Strictly merge Booth group/cycle shards without averaging ratios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any

from .build_frame_lists import read_frame_ids
from .group_counter import GroupCounts
from .report_group_cycle import (
    generate_reports,
    load_aggregate_records,
    write_aggregate_records,
)


KEY_FIELDS = (
    "consumer_layer",
    "activation_id",
    "scale_method",
    "mapping_mode",
    "conv_transpose_mapping",
    "padding_policy",
    "pfn_token_policy",
    "N_main",
)


def _key(record: dict[str, Any]) -> tuple:
    return tuple(record.get(field, "") for field in KEY_FIELDS)


def _merge_aggregate_records(shard_dirs: list[Path]) -> list[dict[str, Any]]:
    merged: dict[tuple, dict[str, Any]] = {}
    counters: dict[tuple, GroupCounts] = {}
    for shard in shard_dirs:
        for record in load_aggregate_records(shard / "aggregate_group_counts.json"):
            key = _key(record)
            counts = GroupCounts.from_dict(record["counts"])
            if key not in merged:
                merged[key] = {name: value for name, value in record.items() if name != "counts"}
                counters[key] = GroupCounts(
                    group_size=counts.group_size,
                    window_sizes=counts.window_sizes,
                    sensitivity_retained=counts.sensitivity_retained,
                )
            elif any(merged[key].get(name) != value for name, value in record.items() if name != "counts"):
                raise ValueError(f"aggregate metadata mismatch for {key}")
            counters[key].add(counts)
    records = []
    for key in sorted(merged, key=lambda item: (merged[item]["layer_order"], merged[item]["scale_method"])):
        records.append({**merged[key], "counts": counters[key].to_dict()})
    return records


def _merge_per_frame_csv(shard_dirs: list[Path], destination: Path) -> int:
    fieldnames: list[str] | None = None
    row_count = 0
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = None
        for shard in shard_dirs:
            with (shard / "per_frame_consumer_counts.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                elif list(reader.fieldnames or []) != fieldnames:
                    raise ValueError(f"per-frame CSV schema mismatch in {shard}")
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
    if not fieldnames or row_count == 0:
        raise ValueError("no per-frame rows were merged")
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-output-parallelism", type=int)
    parser.add_argument("--correction-output-parallelism", type=int)
    parser.add_argument("--scalar-correction-parallelism", type=int, default=8)
    parser.add_argument("--metrics-mode", choices=("full", "core"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    shard_dirs = sorted(
        path for path in args.shards_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not shard_dirs:
        raise ValueError("no shard directories found")

    expected_ids = read_frame_ids(args.manifest)
    processed_ids: list[str] = []
    reference_hashes: dict[str, str] | None = None
    reference_group_config: bytes | None = None
    reference_registry: bytes | None = None
    for shard in shard_dirs:
        checks = json.loads((shard / "validation_checks.json").read_text(encoding="utf-8"))
        if checks.get("status") != "complete" or not checks.get("frame_order_exact"):
            raise ValueError(f"incomplete or unordered shard: {shard}")
        hashes = {
            key: checks[key]
            for key in (
                "manifest_sha256", "scales_sha256", "checkpoint_sha256", "config_sha256",
                "group_cycle_config_sha256",
            )
        }
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise ValueError(f"input hash mismatch in shard: {shard}")
        group_config = (shard / "resolved_group_cycle_config.json").read_bytes()
        registry = (shard / "activation_registry.json").read_bytes()
        if reference_group_config is None:
            reference_group_config = group_config
            reference_registry = registry
        elif group_config != reference_group_config or registry != reference_registry:
            raise ValueError(f"mapping config or registry mismatch in shard: {shard}")
        processed_ids.extend(checks["processed_frame_ids"])
    if processed_ids != expected_ids:
        raise ValueError("merged shard frame IDs do not exactly match the manifest")

    args.output_dir.mkdir(parents=True)
    shutil.copy2(
        shard_dirs[0] / "resolved_group_cycle_config.json",
        args.output_dir / "resolved_group_cycle_config.json",
    )
    shutil.copy2(
        shard_dirs[0] / "activation_registry.json",
        args.output_dir / "activation_registry.json",
    )
    records = _merge_aggregate_records(shard_dirs)
    detected_metrics_mode = json.loads(reference_group_config or b"{}").get(
        "metrics_mode", "full"
    )
    if args.metrics_mode is not None and args.metrics_mode != detected_metrics_mode:
        raise ValueError(
            f"requested metrics mode {args.metrics_mode!r} does not match shards "
            f"{detected_metrics_mode!r}"
        )
    write_aggregate_records(args.output_dir / "aggregate_group_counts.json", records)
    per_frame_rows = _merge_per_frame_csv(
        shard_dirs, args.output_dir / "per_frame_consumer_counts.csv"
    )
    report_summary = generate_reports(
        records,
        args.output_dir,
        main_output_parallelism=args.main_output_parallelism,
        correction_output_parallelism=args.correction_output_parallelism,
        scalar_correction_parallelism=args.scalar_correction_parallelism,
        include_sensitivity=detected_metrics_mode == "full",
    )
    checks = {
        "status": "complete",
        "shard_count": len(shard_dirs),
        "processed_frame_count": len(processed_ids),
        "unique_frame_count": len(set(processed_ids)),
        "frame_order_exact": processed_ids == expected_ids,
        "aggregate_record_count": len(records),
        "per_frame_record_count": per_frame_rows,
        "all_group_counter_invariants_passed": True,
        "metrics_mode": detected_metrics_mode,
        "input_hashes": reference_hashes,
        "report_summary": report_summary,
    }
    (args.output_dir / "validation_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    main()
