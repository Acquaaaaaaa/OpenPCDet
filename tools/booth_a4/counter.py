"""Exact integer counters for activation-level Booth A4 profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from .booth_radix4 import a4_effective_mask, encode_modified_radix4
from .quantizer import quantize_symmetric_int8


_INT64_MAX = 2**63 - 1


def _zero_histogram() -> list[int]:
    return [0] * 256


def _zero_digit_histogram() -> list[list[int]]:
    return [[0] * 5 for _ in range(4)]


@dataclass
class BoothA4Counts:
    n_total: int = 0
    n_a4: int = 0
    n_exception: int = 0
    n_zero_fp32_exact: int = 0
    n_zero_quantized: int = 0
    n_zero_from_rounding: int = 0
    n_nonzero_quantized: int = 0
    n_a4_nonzero_quantized: int = 0
    n_clipped_low: int = 0
    n_clipped_high: int = 0
    n_q_minus128: int = 0
    n_booth_oracle_conflict: int = 0
    call_count: int = 0
    empty_call_count: int = 0
    q_min: int | None = None
    q_max: int | None = None
    int8_hist: list[int] = field(default_factory=_zero_histogram)
    digit_hist: list[list[int]] = field(default_factory=_zero_digit_histogram)

    def validate(self) -> None:
        scalar_counts = (
            self.n_total,
            self.n_a4,
            self.n_exception,
            self.n_zero_fp32_exact,
            self.n_zero_quantized,
            self.n_zero_from_rounding,
            self.n_nonzero_quantized,
            self.n_a4_nonzero_quantized,
            self.n_clipped_low,
            self.n_clipped_high,
            self.n_q_minus128,
            self.n_booth_oracle_conflict,
            self.call_count,
            self.empty_call_count,
        )
        if any(value < 0 or value > _INT64_MAX for value in scalar_counts):
            raise OverflowError("counter is outside non-negative INT64 range")
        if self.n_total != self.n_a4 + self.n_exception:
            raise ValueError("N_total != N_A4 + N_exception")
        if self.n_total != self.n_zero_quantized + self.n_nonzero_quantized:
            raise ValueError("N_total != N_zero_quantized + N_nonzero_quantized")
        if self.n_booth_oracle_conflict != 0:
            raise ValueError("Booth digit predicate conflicts with q in [-8, 7] oracle")
        if len(self.int8_hist) != 256 or sum(self.int8_hist) != self.n_total:
            raise ValueError("INT8 histogram does not sum to N_total")
        if len(self.digit_hist) != 4 or any(len(row) != 5 for row in self.digit_hist):
            raise ValueError("digit histogram must have shape [4][5]")
        if any(sum(row) != self.n_total for row in self.digit_hist):
            raise ValueError("a Booth digit histogram does not sum to N_total")
        if self.n_total == 0 and (self.q_min is not None or self.q_max is not None):
            raise ValueError("empty counts must not define q_min/q_max")
        if self.n_total > 0 and (self.q_min is None or self.q_max is None):
            raise ValueError("non-empty counts must define q_min/q_max")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["RA4_tensor"] = self.n_a4 / self.n_total if self.n_total else None
        result["Rexc_tensor"] = self.n_exception / self.n_total if self.n_total else None
        result["RA4_nonzero_q"] = (
            self.n_a4_nonzero_quantized / self.n_nonzero_quantized
            if self.n_nonzero_quantized
            else None
        )
        return result


def _histogram(values: torch.Tensor, bins: int) -> list[int]:
    return torch.bincount(values.reshape(-1), minlength=bins).to("cpu").tolist()


def count_activation(x: torch.Tensor, scale: float | torch.Tensor) -> BoothA4Counts:
    """Quantize one FP32 reference activation and return exact base counts."""

    quantized = quantize_symmetric_int8(x, scale)
    q = quantized.q
    n_total = q.numel()
    if n_total == 0:
        counts = BoothA4Counts(call_count=1, empty_call_count=1)
        counts.validate()
        return counts

    digits = encode_modified_radix4(q)
    a4_mask = a4_effective_mask(digits)
    q_wide = q.to(torch.int16)
    zero_q = q_wide == 0
    zero_fp = x == 0
    nonzero_q = ~zero_q

    int8_hist = _histogram((q_wide + 128).to(torch.long), 256)
    digit_hist = [
        _histogram((digits[..., index].to(torch.int16) + 2).to(torch.long), 5)
        for index in range(4)
    ]
    counts = BoothA4Counts(
        n_total=n_total,
        n_a4=int(a4_mask.sum().item()),
        n_exception=int((~a4_mask).sum().item()),
        n_zero_fp32_exact=int(zero_fp.sum().item()),
        n_zero_quantized=int(zero_q.sum().item()),
        n_zero_from_rounding=int(((~zero_fp) & zero_q).sum().item()),
        n_nonzero_quantized=int(nonzero_q.sum().item()),
        n_a4_nonzero_quantized=int((a4_mask & nonzero_q).sum().item()),
        n_clipped_low=int(quantized.clipped_low.sum().item()),
        n_clipped_high=int(quantized.clipped_high.sum().item()),
        n_q_minus128=int((q_wide == -128).sum().item()),
        n_booth_oracle_conflict=int(
            (a4_mask != ((q_wide >= -8) & (q_wide <= 7))).sum().item()
        ),
        call_count=1,
        q_min=int(q_wide.min().item()),
        q_max=int(q_wide.max().item()),
        int8_hist=int8_hist,
        digit_hist=digit_hist,
    )
    counts.validate()
    return counts


class BoothA4Accumulator:
    """Merge call- or shard-level integer counts without averaging ratios."""

    def __init__(self) -> None:
        self.counts = BoothA4Counts()

    def add(self, other: BoothA4Counts) -> None:
        other.validate()
        current = self.counts
        for name in (
            "n_total",
            "n_a4",
            "n_exception",
            "n_zero_fp32_exact",
            "n_zero_quantized",
            "n_zero_from_rounding",
            "n_nonzero_quantized",
            "n_a4_nonzero_quantized",
            "n_clipped_low",
            "n_clipped_high",
            "n_q_minus128",
            "n_booth_oracle_conflict",
            "call_count",
            "empty_call_count",
        ):
            setattr(current, name, getattr(current, name) + getattr(other, name))
        current.int8_hist = [a + b for a, b in zip(current.int8_hist, other.int8_hist)]
        current.digit_hist = [
            [a + b for a, b in zip(left, right)]
            for left, right in zip(current.digit_hist, other.digit_hist)
        ]
        if other.q_min is not None:
            current.q_min = other.q_min if current.q_min is None else min(current.q_min, other.q_min)
            current.q_max = other.q_max if current.q_max is None else max(current.q_max, other.q_max)
        current.validate()

    def to_dict(self) -> dict[str, Any]:
        return self.counts.to_dict()


def count_activation_chunked(
    x: torch.Tensor,
    scale: float | torch.Tensor,
    *,
    chunk_elements: int = 4_000_000,
) -> BoothA4Counts:
    """Count one logical activation call in bounded-memory flat chunks."""

    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    if x.numel() == 0:
        return count_activation(x, scale)
    flattened = x.reshape(-1)
    accumulator = BoothA4Accumulator()
    for start in range(0, flattened.numel(), chunk_elements):
        accumulator.add(count_activation(flattened[start : start + chunk_elements], scale))
    result = accumulator.counts
    result.call_count = 1
    result.empty_call_count = 0
    result.validate()
    return result
