"""Exact integer counters for consumer-aware Booth CIM groups."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import torch

from .operand_mapper import OperandGroupBatch


SERIES_NAMES = (
    "mixed",
    "d2",
    "d3",
    "fused",
    "weighted",
    "weighted_d2",
    "weighted_d3",
)

CORE_SERIES_NAMES = ("mixed", "d2", "d3")


def _zeros(length: int) -> list[int]:
    return [0] * length


def _digit_histograms(group_size: int) -> list[list[int]]:
    return [[0] * (group_size + 1) for _ in range(4)]


def _digit_values() -> list[list[int]]:
    return [[0] * 5 for _ in range(4)]


def _empty_schedule_histograms(window_sizes: Iterable[int]) -> dict[str, dict[str, dict[int, int]]]:
    domains = ["per_group", *(f"window_{size}" for size in window_sizes), "layer_frame"]
    return {
        domain: {series: {} for series in SERIES_NAMES}
        for domain in domains
    }


def _add_sparse_histogram(destination: dict[int, int], values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    bins = torch.bincount(values.to(torch.int64))
    occupied = torch.nonzero(bins, as_tuple=False).flatten()
    if occupied.numel() == 0:
        return
    unique = occupied.detach().to(device="cpu").tolist()
    counts = bins[occupied].detach().to(device="cpu").tolist()
    for value, count in zip(unique, counts):
        destination[int(value)] = destination.get(int(value), 0) + int(count)


@dataclass
class GroupCounts:
    group_size: int
    window_sizes: tuple[int, ...] = (8, 32, 128)
    sensitivity_retained: bool = True
    n_group_total: int = 0
    n_tail_group: int = 0
    n_tail_padding_rows: int = 0
    n_boundary_padding_rows: int = 0
    n_zero_insertion_rows: int = 0
    n_pfn_invalid_slot_rows: int = 0
    n_logical_valid_rows: int = 0
    n_nonzero_quantized: int = 0
    n_nonzero_quantized_logical_valid: int = 0
    n_active_group: list[int] = field(default_factory=lambda: _zeros(4))
    n_nonzero_digit: list[int] = field(default_factory=lambda: _zeros(4))
    n_nonzero_digit_logical_valid: list[int] = field(default_factory=lambda: _zeros(4))
    hist_k: list[list[int]] = field(default_factory=list)
    digit_value_counts: list[list[int]] = field(default_factory=_digit_values)
    h_d2_or_d3: int = 0
    h_d2_and_d3: int = 0
    schedule_histograms: dict[str, dict[str, dict[int, int]]] = field(default_factory=dict)
    finalized: bool = False

    def __post_init__(self) -> None:
        self.window_sizes = tuple(int(value) for value in self.window_sizes)
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if any(value <= 0 for value in self.window_sizes):
            raise ValueError("window sizes must be positive")
        if not self.hist_k:
            self.hist_k = _digit_histograms(self.group_size)
        if not self.schedule_histograms:
            self.schedule_histograms = _empty_schedule_histograms(self.window_sizes)

    @property
    def n_physical_rows(self) -> int:
        return self.group_size * self.n_group_total

    @property
    def h_d2(self) -> int:
        return self.n_nonzero_digit[2]

    @property
    def h_d3(self) -> int:
        return self.n_nonzero_digit[3]

    @property
    def h_total(self) -> int:
        return self.h_d2 + self.h_d3

    def validate(self) -> None:
        if len(self.n_active_group) != 4 or len(self.n_nonzero_digit) != 4:
            raise ValueError("digit counters must have length four")
        if len(self.hist_k) != 4 or any(
            len(histogram) != self.group_size + 1 for histogram in self.hist_k
        ):
            raise ValueError("hist_k has an invalid shape")
        if len(self.digit_value_counts) != 4 or any(
            len(histogram) != 5 for histogram in self.digit_value_counts
        ):
            raise ValueError("digit_value_counts has an invalid shape")
        classified = (
            self.n_logical_valid_rows
            + self.n_boundary_padding_rows
            + self.n_zero_insertion_rows
            + self.n_tail_padding_rows
            + self.n_pfn_invalid_slot_rows
        )
        if classified != self.n_physical_rows:
            raise ValueError("physical row classifications do not sum to N*groups")
        for digit in range(4):
            if sum(self.hist_k[digit]) != self.n_group_total:
                raise ValueError("a K histogram does not sum to the group count")
            if self.n_active_group[digit] != self.n_group_total - self.hist_k[digit][0]:
                raise ValueError("active-group count conflicts with hist_K[0]")
            if self.n_nonzero_digit[digit] != sum(
                occupancy * count for occupancy, count in enumerate(self.hist_k[digit])
            ):
                raise ValueError("nonzero-digit count conflicts with hist_K")
            if sum(self.digit_value_counts[digit]) != self.n_physical_rows:
                raise ValueError("digit value histogram does not sum to physical rows")
            if self.n_nonzero_digit_logical_valid[digit] > self.n_logical_valid_rows:
                raise ValueError("valid nonzero digit count exceeds logical-valid rows")
        if self.h_d2_or_d3 < self.h_d2_and_d3:
            raise ValueError("d2-or-d3 count is smaller than d2-and-d3 count")
        if self.n_nonzero_quantized_logical_valid > self.n_logical_valid_rows:
            raise ValueError("valid nonzero activation count exceeds valid rows")
        expected_domains = {
            "per_group", *(f"window_{size}" for size in self.window_sizes), "layer_frame"
        }
        if set(self.schedule_histograms) != expected_domains:
            raise ValueError("schedule histogram domains are incomplete")
        for series_by_name in self.schedule_histograms.values():
            if set(series_by_name) != set(SERIES_NAMES):
                raise ValueError("schedule histogram series are incomplete")
            for histogram in series_by_name.values():
                if any(int(items) < 0 or int(units) < 0 for items, units in histogram.items()):
                    raise ValueError("schedule histograms must be non-negative")

    def add(self, other: "GroupCounts") -> None:
        if (
            self.group_size != other.group_size
            or self.window_sizes != other.window_sizes
            or self.sensitivity_retained != other.sensitivity_retained
        ):
            raise ValueError("cannot merge counters with different group/window configurations")
        if not other.finalized:
            raise ValueError("cannot merge a non-finalized counter")
        scalar_fields = (
            "n_group_total",
            "n_tail_group",
            "n_tail_padding_rows",
            "n_boundary_padding_rows",
            "n_zero_insertion_rows",
            "n_pfn_invalid_slot_rows",
            "n_logical_valid_rows",
            "n_nonzero_quantized",
            "n_nonzero_quantized_logical_valid",
            "h_d2_or_d3",
            "h_d2_and_d3",
        )
        for name in scalar_fields:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for field_name in (
            "n_active_group", "n_nonzero_digit", "n_nonzero_digit_logical_valid"
        ):
            left = getattr(self, field_name)
            right = getattr(other, field_name)
            setattr(self, field_name, [a + b for a, b in zip(left, right)])
        self.hist_k = [
            [a + b for a, b in zip(left, right)]
            for left, right in zip(self.hist_k, other.hist_k)
        ]
        self.digit_value_counts = [
            [a + b for a, b in zip(left, right)]
            for left, right in zip(self.digit_value_counts, other.digit_value_counts)
        ]
        for domain, series_by_name in other.schedule_histograms.items():
            for series, histogram in series_by_name.items():
                destination = self.schedule_histograms[domain][series]
                for items, unit_count in histogram.items():
                    item_key = int(items)
                    destination[item_key] = destination.get(item_key, 0) + int(unit_count)
        self.finalized = True
        self.validate()

    def metrics(self) -> dict[str, Any]:
        self.validate()
        result: dict[str, Any] = {
            "N_main": self.group_size,
            "N_group_total": self.n_group_total,
            "N_physical_rows": self.n_physical_rows,
            "N_logical_valid_rows": self.n_logical_valid_rows,
            "N_boundary_padding_rows": self.n_boundary_padding_rows,
            "N_zero_insertion_rows": self.n_zero_insertion_rows,
            "N_tail_padding_rows": self.n_tail_padding_rows,
            "N_pfn_invalid_slot_rows": self.n_pfn_invalid_slot_rows,
            "N_nonzero_quantized": self.n_nonzero_quantized,
            "N_nonzero_quantized_logical_valid": self.n_nonzero_quantized_logical_valid,
            "H_d2": self.h_d2,
            "H_d3": self.h_d3,
            "H_total": self.h_total,
            "H_d2_or_d3": self.h_d2_or_d3,
            "H_d2_and_d3": self.h_d2_and_d3,
        }
        for digit in range(4):
            active = self.n_active_group[digit]
            nonzero = self.n_nonzero_digit[digit]
            result[f"N_active_group_d{digit}"] = active
            result[f"N_nonzero_digit_d{digit}"] = nonzero
            result[f"P_active_d{digit}"] = (
                active / self.n_group_total if self.n_group_total else None
            )
            result[f"P_skip_d{digit}"] = (
                1.0 - result[f"P_active_d{digit}"]
                if result[f"P_active_d{digit}"] is not None else None
            )
            result[f"U_active_d{digit}"] = (
                nonzero / (self.group_size * active) if active else None
            )
            result[f"U_all_d{digit}"] = (
                nonzero / self.n_physical_rows if self.n_physical_rows else None
            )
            result[f"D_valid_d{digit}"] = (
                self.n_nonzero_digit_logical_valid[digit] / self.n_logical_valid_rows
                if self.n_logical_valid_rows else None
            )
            result[f"D_nonzero_q_d{digit}"] = (
                self.n_nonzero_digit_logical_valid[digit]
                / self.n_nonzero_quantized_logical_valid
                if self.n_nonzero_quantized_logical_valid else None
            )
            if active:
                if not (1 / self.group_size <= result[f"U_active_d{digit}"] <= 1):
                    raise ValueError("active-cycle utilization is outside its valid range")
        return result

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["schedule_histograms"] = {
            domain: {
                series: {str(items): count for items, count in histogram.items()}
                for series, histogram in series_by_name.items()
            }
            for domain, series_by_name in self.schedule_histograms.items()
        }
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroupCounts":
        data = dict(payload)
        data["window_sizes"] = tuple(data["window_sizes"])
        data["schedule_histograms"] = {
            domain: {
                series: {int(items): int(count) for items, count in histogram.items()}
                for series, histogram in series_by_name.items()
            }
            for domain, series_by_name in data["schedule_histograms"].items()
        }
        result = cls(**data)
        result.validate()
        return result


class GroupCounter:
    """Stream group batches into exact counts and schedule-domain histograms."""

    def __init__(
        self,
        group_size: int = 8,
        *,
        window_sizes: Iterable[int] = (8, 32, 128),
        retain_sensitivity: bool = True,
    ):
        effective_windows = tuple(window_sizes) if retain_sensitivity else ()
        self.counts = GroupCounts(
            group_size=group_size,
            window_sizes=effective_windows,
            sensitivity_retained=retain_sensitivity,
        )
        self._series_names = SERIES_NAMES if retain_sensitivity else CORE_SERIES_NAMES
        self._window_carry: dict[int, dict[str, torch.Tensor | None]] = {
            size: {
                series: None
                for series in self._series_names
            }
            for size in self.counts.window_sizes
        }
        self._series_totals = {series: 0 for series in self._series_names}

    @staticmethod
    def _series(
        digits: torch.Tensor, *, retain_sensitivity: bool
    ) -> dict[str, torch.Tensor]:
        d2 = digits[:, :, 2]
        d3 = digits[:, :, 3]
        nz2 = d2 != 0
        nz3 = d3 != 0
        count_d2 = nz2.sum(dim=1, dtype=torch.int64)
        count_d3 = nz3.sum(dim=1, dtype=torch.int64)
        result = {
            "mixed": count_d2 + count_d3,
            "d2": count_d2,
            "d3": count_d3,
        }
        if not retain_sensitivity:
            return result
        abs2 = d2.abs()
        abs3 = d3.abs()
        weighted_d2 = ((abs2 == 1).to(torch.int64) + 2 * (abs2 == 2)).sum(dim=1)
        weighted_d3 = ((abs3 == 1).to(torch.int64) + 2 * (abs3 == 2)).sum(dim=1)
        result.update({
            "fused": (nz2 | nz3).sum(dim=1, dtype=torch.int64),
            "weighted": weighted_d2 + weighted_d3,
            "weighted_d2": weighted_d2,
            "weighted_d3": weighted_d3,
        })
        return result

    def _update_schedule_histograms(self, series_values: dict[str, torch.Tensor]) -> None:
        for series, values in series_values.items():
            values = values.detach().to(dtype=torch.int64)
            if self.counts.sensitivity_retained:
                _add_sparse_histogram(
                    self.counts.schedule_histograms["per_group"][series], values
                )
            self._series_totals[series] += int(values.sum().item())
            for window_size in self.counts.window_sizes:
                carry = self._window_carry[window_size][series]
                combined = values if carry is None else torch.cat((carry, values))
                complete_length = (combined.numel() // window_size) * window_size
                if complete_length:
                    window_sums = combined[:complete_length].reshape(-1, window_size).sum(dim=1)
                    _add_sparse_histogram(
                        self.counts.schedule_histograms[f"window_{window_size}"][series],
                        window_sums,
                    )
                self._window_carry[window_size][series] = combined[complete_length:]

    def update(self, batch: OperandGroupBatch) -> None:
        if self.counts.finalized:
            raise RuntimeError("cannot update a finalized counter")
        batch.validate(self.counts.group_size)
        digits = batch.digits
        group_count = digits.shape[0]
        nonzero = digits != 0
        occupancy = nonzero.sum(dim=1, dtype=torch.int64)
        logical = batch.logical_valid
        quantized_nonzero = nonzero.any(dim=2)

        self.counts.n_group_total += group_count
        self.counts.n_tail_group += int(batch.tail_padding.any(dim=1).sum().item())
        self.counts.n_tail_padding_rows += int(batch.tail_padding.sum().item())
        self.counts.n_boundary_padding_rows += int(batch.boundary_padding.sum().item())
        self.counts.n_zero_insertion_rows += int(batch.zero_insertion_mask.sum().item())
        self.counts.n_pfn_invalid_slot_rows += int(batch.pfn_invalid_slot.sum().item())
        self.counts.n_logical_valid_rows += int(logical.sum().item())
        self.counts.n_nonzero_quantized += int(quantized_nonzero.sum().item())
        self.counts.n_nonzero_quantized_logical_valid += int(
            (quantized_nonzero & logical).sum().item()
        )

        for digit in range(4):
            current = occupancy[:, digit]
            histogram = torch.bincount(
                current.detach(), minlength=self.counts.group_size + 1
            ).to(device="cpu").tolist()
            self.counts.hist_k[digit] = [
                a + int(b) for a, b in zip(self.counts.hist_k[digit], histogram)
            ]
            self.counts.n_active_group[digit] += group_count - int(histogram[0])
            self.counts.n_nonzero_digit[digit] += sum(
                occupancy_value * int(count)
                for occupancy_value, count in enumerate(histogram)
            )
            self.counts.n_nonzero_digit_logical_valid[digit] += int(
                (nonzero[:, :, digit] & logical).sum().item()
            )
            values = (digits[:, :, digit].to(torch.int16) + 2).reshape(-1)
            value_histogram = torch.bincount(
                values.detach().to(dtype=torch.int64), minlength=5
            ).to(device="cpu").tolist()
            self.counts.digit_value_counts[digit] = [
                a + int(b)
                for a, b in zip(self.counts.digit_value_counts[digit], value_histogram)
            ]

        nz2 = nonzero[:, :, 2]
        nz3 = nonzero[:, :, 3]
        self.counts.h_d2_or_d3 += int((nz2 | nz3).sum().item())
        self.counts.h_d2_and_d3 += int((nz2 & nz3).sum().item())
        self._update_schedule_histograms(
            self._series(digits, retain_sensitivity=self.counts.sensitivity_retained)
        )

    def finalize(self) -> GroupCounts:
        if self.counts.finalized:
            return self.counts
        for window_size in self.counts.window_sizes:
            domain = self.counts.schedule_histograms[f"window_{window_size}"]
            for series in self._series_names:
                carry = self._window_carry[window_size][series]
                if carry is not None and carry.numel():
                    items = int(carry.sum().item())
                    domain[series][items] = domain[series].get(items, 0) + 1
        layer_frame = self.counts.schedule_histograms["layer_frame"]
        for series, items in self._series_totals.items():
            layer_frame[series][items] = layer_frame[series].get(items, 0) + 1
        self.counts.finalized = True
        self.counts.validate()
        return self.counts


def count_group_batches(
    batches: Iterable[OperandGroupBatch],
    *,
    group_size: int = 8,
    window_sizes: Iterable[int] = (8, 32, 128),
    retain_sensitivity: bool = True,
) -> GroupCounts:
    counter = GroupCounter(
        group_size,
        window_sizes=window_sizes,
        retain_sensitivity=retain_sensitivity,
    )
    for batch in batches:
        counter.update(batch)
    return counter.finalize()
