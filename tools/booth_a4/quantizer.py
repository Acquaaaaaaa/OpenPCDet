"""Frozen per-tensor signed-symmetric INT8 reference quantizer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Int8QuantizationResult:
    """Integer codes and pre-clamp diagnostics for one activation tensor."""

    q: torch.Tensor
    q_preclamp: torch.Tensor
    clipped_low: torch.Tensor
    clipped_high: torch.Tensor


def _validate_scale(scale: float | torch.Tensor) -> float:
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("per-tensor scale must be scalar")
        scale_value = float(scale.detach().cpu().item())
    else:
        scale_value = float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale_value!r}")
    return scale_value


def quantize_symmetric_int8(
    x: torch.Tensor,
    scale: float | torch.Tensor,
) -> Int8QuantizationResult:
    """Map FP activation values to ``[-128, 127]`` with nearest-even rounding.

    This is an independent reference mapping. It does not feed dequantized
    values back into the model and therefore does not propagate quantization
    error to downstream layers.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.is_complex():
        raise TypeError("complex activation tensors are not supported")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("activation contains NaN or Inf")

    scale_value = _validate_scale(scale)
    q_preclamp = torch.round(x / scale_value).to(torch.int64)
    clipped_low = q_preclamp < -128
    clipped_high = q_preclamp > 127
    q = torch.clamp(q_preclamp, -128, 127).to(torch.int8)
    return Int8QuantizationResult(
        q=q,
        q_preclamp=q_preclamp,
        clipped_low=clipped_low,
        clipped_high=clipped_high,
    )


def scale_from_threshold(threshold: float) -> float:
    """Derive the guide's signed INT8 scale ``threshold / 127``."""

    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or threshold_value <= 0.0:
        raise ValueError(f"threshold must be finite and positive, got {threshold_value!r}")
    return threshold_value / 127.0
