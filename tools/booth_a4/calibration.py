"""Streaming activation-distribution collection and frozen scale derivation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

import torch

from .quantizer import scale_from_threshold


SAMPLE_ALGORITHM = "deterministic_stratified_per_call_v1"


@dataclass
class DistributionSummary:
    activation_id: str
    element_count: int = 0
    finite_count: int = 0
    exact_zero_count: int = 0
    nan_count: int = 0
    inf_count: int = 0
    call_count: int = 0
    empty_call_count: int = 0
    float_min: float | None = None
    float_max: float | None = None
    max_abs: float | None = None
    min_positive_abs: float | None = None
    sample_count: int = 0
    sample_cap: int = 1_000_000
    sample_seed: int = 666
    sample_algorithm: str = SAMPLE_ALGORITHM

    def validate(self) -> None:
        if self.element_count != self.finite_count + self.nan_count + self.inf_count:
            raise ValueError("element_count != finite_count + nan_count + inf_count")
        if self.sample_count > self.sample_cap:
            raise ValueError("sample_count exceeds sample_cap")
        if self.finite_count == 0 and any(
            value is not None for value in (self.float_min, self.float_max, self.max_abs)
        ):
            raise ValueError("empty finite population cannot have extrema")


class DistributionAccumulator:
    """Accumulate exact extrema/counts and bounded deterministic abs samples."""

    def __init__(
        self,
        activation_id: str,
        *,
        sample_cap: int = 1_000_000,
        samples_per_call: int = 3906,
        sample_seed: int = 666,
    ) -> None:
        if sample_cap <= 0 or samples_per_call <= 0:
            raise ValueError("sample_cap and samples_per_call must be positive")
        self.summary = DistributionSummary(
            activation_id=activation_id,
            sample_cap=sample_cap,
            sample_seed=sample_seed,
        )
        self.samples_per_call = samples_per_call
        self.sample_chunks: list[torch.Tensor] = []

    def _offset(self, call_index: int) -> float:
        payload = (
            f"{self.summary.sample_seed}:{self.summary.activation_id}:{call_index}"
        ).encode("utf-8")
        integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return integer / 2**64

    def update(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("activation must be a torch.Tensor")
        if tensor.is_complex():
            raise TypeError("complex activation tensors are unsupported")
        summary = self.summary
        summary.call_count += 1
        summary.element_count += tensor.numel()
        if tensor.numel() == 0:
            summary.empty_call_count += 1
            summary.validate()
            return

        detached = tensor.detach()
        nan_count = int(torch.isnan(detached).sum().item()) if detached.is_floating_point() else 0
        inf_count = int(torch.isinf(detached).sum().item()) if detached.is_floating_point() else 0
        summary.nan_count += nan_count
        summary.inf_count += inf_count
        summary.finite_count += tensor.numel() - nan_count - inf_count
        if nan_count or inf_count:
            summary.validate()
            raise FloatingPointError(
                f"{summary.activation_id} contains NaN={nan_count}, Inf={inf_count}"
            )

        values = detached.float()
        current_min, current_max = torch.aminmax(values)
        minimum = float(current_min.item())
        maximum = float(current_max.item())
        summary.float_min = minimum if summary.float_min is None else min(summary.float_min, minimum)
        summary.float_max = maximum if summary.float_max is None else max(summary.float_max, maximum)
        current_max_abs = max(abs(minimum), abs(maximum))
        summary.max_abs = (
            current_max_abs if summary.max_abs is None else max(summary.max_abs, current_max_abs)
        )
        summary.exact_zero_count += int((values == 0).sum().item())
        positive = values.abs()
        positive = positive[positive > 0]
        if positive.numel():
            current_min_positive = float(positive.min().item())
            summary.min_positive_abs = (
                current_min_positive
                if summary.min_positive_abs is None
                else min(summary.min_positive_abs, current_min_positive)
            )

        remaining = summary.sample_cap - summary.sample_count
        take = min(values.numel(), self.samples_per_call, remaining)
        if take:
            flat = values.abs().reshape(-1)
            offset = self._offset(summary.call_count - 1)
            positions = (
                (torch.arange(take, device=flat.device, dtype=torch.float64) + offset)
                * (flat.numel() / take)
            ).floor().to(torch.long)
            positions.clamp_(max=flat.numel() - 1)
            self.sample_chunks.append(flat[positions].to(device="cpu", dtype=torch.float64))
            summary.sample_count += take
        summary.validate()

    def samples(self) -> torch.Tensor:
        if not self.sample_chunks:
            return torch.empty(0, dtype=torch.float64)
        return torch.cat(self.sample_chunks)

    def to_dict(self) -> dict[str, Any]:
        self.summary.validate()
        result = asdict(self.summary)
        result["all_zero_in_calibration"] = (
            self.summary.finite_count > 0
            and self.summary.exact_zero_count == self.summary.finite_count
        )
        return result


def _method(threshold: float | None, *, all_zero: bool) -> dict[str, Any]:
    if all_zero:
        return {"threshold": None, "scale": None, "status": "uncalibratable_all_zero"}
    if threshold is None or not math.isfinite(threshold) or threshold <= 0:
        return {"threshold": threshold, "scale": None, "status": "invalid_zero_threshold"}
    return {
        "threshold": threshold,
        "scale": scale_from_threshold(threshold),
        "status": "ok",
    }


def derive_activation_scales(
    summary: DistributionSummary,
    samples: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive min-max/P99.99/P99.9 scales with pinned linear interpolation."""

    summary.validate()
    if summary.nan_count or summary.inf_count:
        raise ValueError("cannot derive scales from non-finite activation data")
    if samples.numel() != summary.sample_count:
        raise ValueError("sample tensor length does not match sample_count")
    all_zero = summary.finite_count > 0 and summary.exact_zero_count == summary.finite_count

    percentile_values: dict[str, float | None] = {"p99_99": None, "p99_9": None}
    nonzero_values: dict[str, float | None] = {"p99_99": None, "p99_9": None}
    if samples.numel():
        samples = samples.to(torch.float64)
        for name, probability in (("p99_99", 0.9999), ("p99_9", 0.999)):
            percentile_values[name] = float(
                torch.quantile(samples, probability, interpolation="linear").item()
            )
        nonzero = samples[samples > 0]
        if nonzero.numel():
            for name, probability in (("p99_99", 0.9999), ("p99_9", 0.999)):
                nonzero_values[name] = float(
                    torch.quantile(nonzero, probability, interpolation="linear").item()
                )

    scales = {
        "all_zero_in_calibration": all_zero,
        "calibration_status": "uncalibratable_all_zero" if all_zero else "ok",
        "sample_count": summary.sample_count,
        "sample_cap": summary.sample_cap,
        "sample_seed": summary.sample_seed,
        "sample_algorithm": summary.sample_algorithm,
        "percentile_interpolation": "linear",
        "percentile_compute_dtype": "float64",
        "minmax": _method(summary.max_abs, all_zero=all_zero),
        "p99_99": _method(percentile_values["p99_99"], all_zero=all_zero),
        "p99_9": _method(percentile_values["p99_9"], all_zero=all_zero),
    }
    diagnostics = {
        "sample_zero_count": int((samples == 0).sum().item()),
        "sample_nonzero_count": int((samples > 0).sum().item()),
        "all_value_percentiles": percentile_values,
        "nonzero_only_percentiles": nonzero_values,
    }
    return scales, diagnostics
