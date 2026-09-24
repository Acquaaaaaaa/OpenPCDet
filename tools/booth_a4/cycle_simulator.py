"""Normalized and output-aware cycle models for Booth group execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .group_counter import GroupCounts


@dataclass(frozen=True)
class CycleConfig:
    correction_parallelism: int = 8
    correction_schedule_domain: str = "layer_frame"
    correction_plane_queue: str = "mixed"
    correction_item_mode: str = "nonzero_digit"
    timing_mode: str = "serial"
    main_output_parallelism: int | None = None
    correction_output_parallelism: int | None = None
    scalar_correction_parallelism: int = 8

    def validate(self) -> None:
        if self.correction_parallelism <= 0 or self.scalar_correction_parallelism <= 0:
            raise ValueError("correction parallelism values must be positive")
        if self.main_output_parallelism is not None and self.main_output_parallelism <= 0:
            raise ValueError("main_output_parallelism must be positive when specified")
        if (
            self.correction_output_parallelism is not None
            and self.correction_output_parallelism <= 0
        ):
            raise ValueError("correction_output_parallelism must be positive when specified")
        if self.correction_plane_queue not in {"mixed", "separate"}:
            raise ValueError("correction_plane_queue must be mixed or separate")
        if self.correction_item_mode not in {
            "nonzero_digit", "fused_residual", "magnitude_weighted"
        }:
            raise ValueError("unknown correction_item_mode")
        if self.correction_item_mode == "fused_residual" and self.correction_plane_queue != "mixed":
            raise ValueError("fused residual items require a mixed plane queue")
        if self.timing_mode not in {"serial", "ideal_overlap"}:
            raise ValueError("timing_mode must be serial or ideal_overlap")


def _series_names(config: CycleConfig) -> tuple[str, ...]:
    if config.correction_item_mode == "fused_residual":
        return ("fused",)
    if config.correction_item_mode == "magnitude_weighted":
        return ("weighted",) if config.correction_plane_queue == "mixed" else (
            "weighted_d2", "weighted_d3"
        )
    return ("mixed",) if config.correction_plane_queue == "mixed" else ("d2", "d3")


def _domain_name(config: CycleConfig) -> str:
    domain = config.correction_schedule_domain
    if domain in {"per_group", "layer_frame"}:
        return domain
    if domain.startswith("window_"):
        return domain
    raise ValueError(f"unknown correction schedule domain: {domain}")


def correction_cycles(
    counts: GroupCounts,
    config: CycleConfig,
    *,
    parallelism: int | None = None,
    item_multiplier: int = 1,
) -> tuple[int, int]:
    """Return cycles and logical item count within the configured queue boundaries."""

    config.validate()
    counts.validate()
    lanes = config.correction_parallelism if parallelism is None else int(parallelism)
    if lanes <= 0 or item_multiplier <= 0:
        raise ValueError("parallelism and item_multiplier must be positive")
    domain = _domain_name(config)
    if domain not in counts.schedule_histograms:
        raise ValueError(
            f"schedule domain {domain!r} was not retained; available: "
            f"{sorted(counts.schedule_histograms)}"
        )
    cycles = 0
    items = 0
    for series in _series_names(config):
        histogram = counts.schedule_histograms[domain][series]
        for item_count, unit_count in histogram.items():
            scaled_items = int(item_count) * item_multiplier
            items += scaled_items * int(unit_count)
            cycles += math.ceil(scaled_items / lanes) * int(unit_count)
    return cycles, items


def _speedup(baseline: int | None, candidate: int | None) -> float | None:
    if baseline is None or candidate in (None, 0):
        return None
    return baseline / candidate


def _combined_cycles(main: int, correction: int, timing_mode: str) -> int:
    if timing_mode == "serial":
        return main + correction
    if timing_mode == "ideal_overlap":
        return max(main, correction)
    raise ValueError(f"unknown timing mode: {timing_mode}")


def simulate_cycles(
    counts: GroupCounts,
    *,
    output_channels: int,
    config: CycleConfig = CycleConfig(),
) -> dict[str, Any]:
    """Simulate A/B/C with vector, scalar, and equal-width correction paths.

    The normalized result is for one complete output-channel tile.  Full-layer
    vector results require both main and correction output parallelism.  A
    scalar correction item is one digit-by-one-output-channel operation.
    """

    config.validate()
    counts.validate()
    if output_channels <= 0:
        raise ValueError("output_channels must be positive")

    groups = counts.n_group_total
    c_a_tile = 4 * groups
    c_b_tile = sum(counts.n_active_group)
    c_b_high_tile = counts.n_active_group[2] + counts.n_active_group[3]
    c_main_tile = 2 * groups
    c_corr_vector_tile, vector_items_tile = correction_cycles(counts, config)
    c_c_vector_tile = _combined_cycles(
        c_main_tile, c_corr_vector_tile, config.timing_mode
    )
    vector_utilization = (
        vector_items_tile / (config.correction_parallelism * c_corr_vector_tile)
        if c_corr_vector_tile else None
    )

    c_corr_scalar_full, scalar_items_full = correction_cycles(
        counts,
        config,
        parallelism=config.scalar_correction_parallelism,
        item_multiplier=output_channels,
    )
    scalar_utilization = (
        scalar_items_full / (config.scalar_correction_parallelism * c_corr_scalar_full)
        if c_corr_scalar_full else None
    )

    result: dict[str, Any] = {
        "cycle_scope_primary": "normalized_one_output_tile",
        "correction_datapath_primary": "vector_tile",
        "N_group_total": groups,
        "N_corr_vector_item_lanes": config.correction_parallelism,
        "P_corr_scalar_ops": config.scalar_correction_parallelism,
        "C_out": output_channels,
        "correction_schedule_domain": config.correction_schedule_domain,
        "correction_plane_queue": config.correction_plane_queue,
        "correction_item_mode": config.correction_item_mode,
        "timing_mode": config.timing_mode,
        "C_A_tile": c_a_tile,
        "C_B_tile": c_b_tile,
        "C_B_high_tile": c_b_high_tile,
        "C_main_tile": c_main_tile,
        "C_corr_vector_tile": c_corr_vector_tile,
        "C_C_vector_tile": c_c_vector_tile,
        "speedup_B_vs_A_tile": _speedup(c_a_tile, c_b_tile),
        "speedup_C_vector_vs_A_tile": _speedup(c_a_tile, c_c_vector_tile),
        "speedup_C_vector_vs_B_tile": _speedup(c_b_tile, c_c_vector_tile),
        "U_corr_vector_tile": vector_utilization,
        "cycle_margin_C_vector_vs_B_tile": c_b_tile - c_c_vector_tile,
        "max_overhead_cycles_vector_tile": max(0, c_b_tile - c_c_vector_tile),
        "max_overhead_cycles_per_vector_item_tile": (
            max(0, c_b_tile - c_c_vector_tile) / vector_items_tile
            if vector_items_tile else None
        ),
        "C_corr_scalar_full": c_corr_scalar_full,
        "scalar_item_count_full": scalar_items_full,
        "U_corr_scalar_full": scalar_utilization,
    }

    main_tiles = (
        math.ceil(output_channels / config.main_output_parallelism)
        if config.main_output_parallelism is not None else None
    )
    correction_tiles = (
        math.ceil(output_channels / config.correction_output_parallelism)
        if config.correction_output_parallelism is not None else None
    )
    result["P_out_main"] = config.main_output_parallelism
    result["P_out_corr_vector"] = config.correction_output_parallelism
    result["T_main"] = main_tiles
    result["T_corr_vector"] = correction_tiles

    if main_tiles is None:
        result.update({
            "C_A_full": None,
            "C_B_full": None,
            "C_main_full": None,
            "C_C_scalar_full": None,
            "speedup_C_scalar_vs_B_full": None,
        })
    else:
        c_a_full = c_a_tile * main_tiles
        c_b_full = c_b_tile * main_tiles
        c_main_full = c_main_tile * main_tiles
        c_c_scalar_full = _combined_cycles(
            c_main_full, c_corr_scalar_full, config.timing_mode
        )
        result.update({
            "C_A_full": c_a_full,
            "C_B_full": c_b_full,
            "C_main_full": c_main_full,
            "C_C_scalar_full": c_c_scalar_full,
            "speedup_C_scalar_vs_A_full": _speedup(c_a_full, c_c_scalar_full),
            "speedup_C_scalar_vs_B_full": _speedup(c_b_full, c_c_scalar_full),
        })

    if main_tiles is None or correction_tiles is None:
        result.update({
            "C_corr_vector_full": None,
            "C_C_vector_full": None,
            "speedup_C_vector_vs_A_full": None,
            "speedup_C_vector_vs_B_full": None,
            "P_corr_equal_resource_scalar_ops": None,
            "C_corr_equal_resource_scalar_full": None,
            "C_C_equal_resource_scalar_full": None,
            "speedup_C_equal_resource_scalar_vs_B_full": None,
        })
    else:
        c_corr_vector_full = correction_tiles * c_corr_vector_tile
        c_c_vector_full = _combined_cycles(
            result["C_main_full"], c_corr_vector_full, config.timing_mode
        )
        equal_resource_parallelism = (
            config.correction_parallelism * config.correction_output_parallelism
        )
        c_corr_equal, _ = correction_cycles(
            counts,
            config,
            parallelism=equal_resource_parallelism,
            item_multiplier=output_channels,
        )
        c_c_equal = _combined_cycles(
            result["C_main_full"], c_corr_equal, config.timing_mode
        )
        result.update({
            "C_corr_vector_full": c_corr_vector_full,
            "C_C_vector_full": c_c_vector_full,
            "speedup_C_vector_vs_A_full": _speedup(result["C_A_full"], c_c_vector_full),
            "speedup_C_vector_vs_B_full": _speedup(result["C_B_full"], c_c_vector_full),
            "P_corr_equal_resource_scalar_ops": equal_resource_parallelism,
            "C_corr_equal_resource_scalar_full": c_corr_equal,
            "C_C_equal_resource_scalar_full": c_c_equal,
            "speedup_C_equal_resource_scalar_vs_B_full": _speedup(
                result["C_B_full"], c_c_equal
            ),
        })
    return result
