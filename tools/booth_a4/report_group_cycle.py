"""Generate group-utilization and cycle-model tables from aggregate counters."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .cycle_simulator import CycleConfig, simulate_cycles
from .group_counter import GroupCounts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_aggregate_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported aggregate counter schema")
    records = payload.get("records", [])
    if not records:
        raise ValueError("aggregate counter file contains no records")
    return records


def write_aggregate_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"schema_version": 1, "records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _series_cycles(counts: GroupCounts, series: str, parallelism: int) -> int:
    histogram = counts.schedule_histograms["layer_frame"][series]
    return sum(
        math.ceil(int(items) / parallelism) * int(unit_count)
        for items, unit_count in histogram.items()
    )


def _base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "layer_order",
            "consumer_layer",
            "activation_id",
            "module_type",
            "weight_shape",
            "scale_method",
            "mapping_mode",
            "conv_transpose_mapping",
            "padding_policy",
            "pfn_token_policy",
            "N_main",
            "C_out",
        )
        if key in record
    }


def _primary_config(
    *,
    main_output_parallelism: int | None,
    correction_output_parallelism: int | None,
    scalar_correction_parallelism: int,
) -> CycleConfig:
    return CycleConfig(
        correction_parallelism=8,
        correction_schedule_domain="layer_frame",
        correction_plane_queue="mixed",
        correction_item_mode="nonzero_digit",
        timing_mode="serial",
        main_output_parallelism=main_output_parallelism,
        correction_output_parallelism=correction_output_parallelism,
        scalar_correction_parallelism=scalar_correction_parallelism,
    )


def build_report_rows(
    records: Iterable[dict[str, Any]],
    *,
    main_output_parallelism: int | None = None,
    correction_output_parallelism: int | None = None,
    scalar_correction_parallelism: int = 8,
    include_sensitivity: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    group_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    base_config = _primary_config(
        main_output_parallelism=main_output_parallelism,
        correction_output_parallelism=correction_output_parallelism,
        scalar_correction_parallelism=scalar_correction_parallelism,
    )

    for record in records:
        counts = GroupCounts.from_dict(record["counts"])
        metadata = _base_metadata(record)
        group_row = {**metadata, **counts.metrics()}
        packed_d2 = _series_cycles(counts, "d2", base_config.correction_parallelism)
        packed_d3 = _series_cycles(counts, "d3", base_config.correction_parallelism)
        group_row["S_scatter_d2"] = (
            counts.n_active_group[2] / packed_d2 if packed_d2 else None
        )
        group_row["S_scatter_d3"] = (
            counts.n_active_group[3] / packed_d3 if packed_d3 else None
        )
        group_rows.append(group_row)

        primary_cycles = simulate_cycles(
            counts, output_channels=int(record["C_out"]), config=base_config
        )
        cycle_rows.append({**metadata, **primary_cycles})

        if not include_sensitivity:
            continue

        variants: list[tuple[str, Any, CycleConfig]] = []
        for parallelism in (1, 2, 4, 8, 16, 32):
            variants.append(("N_corr", parallelism, CycleConfig(
                **{**base_config.__dict__, "correction_parallelism": parallelism}
            )))
        for domain in ("per_group", "window_8", "window_32", "window_128", "layer_frame"):
            if domain in counts.schedule_histograms:
                variants.append(("schedule_domain", domain, CycleConfig(
                    **{**base_config.__dict__, "correction_schedule_domain": domain}
                )))
        for queue in ("mixed", "separate"):
            variants.append(("plane_queue", queue, CycleConfig(
                **{**base_config.__dict__, "correction_plane_queue": queue}
            )))
        for item_mode in ("nonzero_digit", "fused_residual", "magnitude_weighted"):
            variants.append(("item_mode", item_mode, CycleConfig(
                **{**base_config.__dict__, "correction_item_mode": item_mode}
            )))
        for timing in ("serial", "ideal_overlap"):
            variants.append(("timing", timing, CycleConfig(
                **{**base_config.__dict__, "timing_mode": timing}
            )))
        seen: set[tuple[str, str]] = set()
        for factor, value, variant in variants:
            key = (factor, str(value))
            if key in seen:
                continue
            seen.add(key)
            result = simulate_cycles(
                counts, output_channels=int(record["C_out"]), config=variant
            )
            sensitivity_rows.append({
                **metadata,
                "sensitivity_factor": factor,
                "sensitivity_value": value,
                **result,
            })
    return group_rows, cycle_rows, sensitivity_rows


def _sum_optional(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [row.get(key) for row in rows]
    return None if any(value is None for value in values) else sum(int(value) for value in values)


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def build_network_rows(cycle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in cycle_rows:
        grouped.setdefault((row["scale_method"], int(row["N_main"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (scale_method, group_size), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "scale_method": scale_method,
            "N_main": group_size,
            "consumer_count": len(rows),
            "cycle_scope_primary": "normalized_one_output_tile_work_sum",
        }
        for key in (
            "C_A_tile", "C_B_tile", "C_main_tile", "C_corr_vector_tile",
            "C_C_vector_tile", "C_A_full", "C_B_full", "C_main_full",
            "C_corr_vector_full", "C_C_vector_full", "C_corr_scalar_full",
            "C_C_scalar_full", "C_corr_equal_resource_scalar_full",
            "C_C_equal_resource_scalar_full",
        ):
            summary[f"{key}_network"] = _sum_optional(rows, key)
        summary["speedup_C_vector_vs_B_tile_network"] = _ratio(
            summary["C_B_tile_network"], summary["C_C_vector_tile_network"]
        )
        summary["speedup_C_vector_vs_B_full_network"] = _ratio(
            summary["C_B_full_network"], summary["C_C_vector_full_network"]
        )
        summary["speedup_C_scalar_vs_B_full_network"] = _ratio(
            summary["C_B_full_network"], summary["C_C_scalar_full_network"]
        )
        summary["speedup_C_equal_resource_scalar_vs_B_full_network"] = _ratio(
            summary["C_B_full_network"],
            summary["C_C_equal_resource_scalar_full_network"],
        )
        output.append(summary)
    return output


def generate_reports(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    main_output_parallelism: int | None = None,
    correction_output_parallelism: int | None = None,
    scalar_correction_parallelism: int = 8,
    include_sensitivity: bool = True,
) -> dict[str, int]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    group_rows, cycle_rows, sensitivity_rows = build_report_rows(
        records,
        main_output_parallelism=main_output_parallelism,
        correction_output_parallelism=correction_output_parallelism,
        scalar_correction_parallelism=scalar_correction_parallelism,
        include_sensitivity=include_sensitivity,
    )
    network_rows = build_network_rows(cycle_rows)
    _write_csv(destination / "consumer_group_utilization.csv", group_rows)
    _write_csv(destination / "consumer_cycle_summary.csv", cycle_rows)
    _write_csv(destination / "network_cycle_summary.csv", network_rows)
    if sensitivity_rows:
        _write_csv(destination / "correction_sensitivity.csv", sensitivity_rows)
    return {
        "group_rows": len(group_rows),
        "cycle_rows": len(cycle_rows),
        "network_rows": len(network_rows),
        "sensitivity_rows": len(sensitivity_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-counts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-output-parallelism", type=int)
    parser.add_argument("--correction-output-parallelism", type=int)
    parser.add_argument("--scalar-correction-parallelism", type=int, default=8)
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()
    records = load_aggregate_records(args.aggregate_counts)
    summary = generate_reports(
        records,
        args.output_dir,
        main_output_parallelism=args.main_output_parallelism,
        correction_output_parallelism=args.correction_output_parallelism,
        scalar_correction_parallelism=args.scalar_correction_parallelism,
        include_sensitivity=not args.primary_only,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
