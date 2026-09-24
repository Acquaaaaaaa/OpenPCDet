"""Exact modified radix-4 Booth recoding for signed INT8 values."""

from __future__ import annotations

import torch


_RECODE_LUT = (0, 1, 1, 2, -2, -1, -1, 0)


def _as_valid_int8_codes(q: torch.Tensor) -> torch.Tensor:
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"q must be a torch.Tensor, got {type(q).__name__}")
    if q.dtype == torch.bool or q.is_complex():
        raise TypeError(f"q must contain integer INT8 codes, got {q.dtype}")

    if q.is_floating_point():
        if not bool(torch.isfinite(q).all()):
            raise ValueError("q contains NaN or Inf")
        if not bool(torch.eq(q, torch.round(q)).all()):
            raise ValueError("q contains non-integral values")

    q_wide = q.to(torch.int16)
    if q_wide.numel():
        q_min = int(q_wide.min().item())
        q_max = int(q_wide.max().item())
        if q_min < -128 or q_max > 127:
            raise ValueError(f"q is outside signed INT8 range: [{q_min}, {q_max}]")
    return q_wide


def encode_modified_radix4(q: torch.Tensor) -> torch.Tensor:
    """Encode signed INT8 codes as ``[..., d0, d1, d2, d3]``.

    The three-bit groups are ``(q1,q0,0)``, ``(q3,q2,q1)``,
    ``(q5,q4,q3)`` and ``(q7,q6,q5)``. Bit operations are deliberately
    performed at INT16 width so the signed INT8 edge cases are unambiguous.
    """

    q_wide = _as_valid_int8_codes(q)
    bits = torch.bitwise_and(q_wide, 0xFF)
    lut = torch.tensor(_RECODE_LUT, dtype=torch.int8, device=q.device)

    def bit(index: int) -> torch.Tensor:
        return torch.bitwise_and(torch.bitwise_right_shift(bits, index), 1)

    codes = torch.stack(
        (
            bit(1) * 4 + bit(0) * 2,
            bit(3) * 4 + bit(2) * 2 + bit(1),
            bit(5) * 4 + bit(4) * 2 + bit(3),
            bit(7) * 4 + bit(6) * 2 + bit(5),
        ),
        dim=-1,
    ).to(torch.long)
    return lut[codes]


def booth_reconstruct(digits: torch.Tensor) -> torch.Tensor:
    """Reconstruct signed values from ``[..., 4]`` Booth digits."""

    if not isinstance(digits, torch.Tensor):
        raise TypeError(f"digits must be a torch.Tensor, got {type(digits).__name__}")
    if digits.ndim == 0 or digits.shape[-1] != 4:
        raise ValueError(f"digits must have final dimension 4, got {tuple(digits.shape)}")
    digits_wide = digits.to(torch.int16)
    if digits_wide.numel() and not bool(((digits_wide >= -2) & (digits_wide <= 2)).all()):
        raise ValueError("digits must be in {-2, -1, 0, 1, 2}")
    weights = torch.tensor((1, 4, 16, 64), dtype=torch.int16, device=digits.device)
    return torch.sum(digits_wide * weights, dim=-1)


def a4_effective_mask(digits: torch.Tensor) -> torch.Tensor:
    """Return the formal A4 predicate ``(d2 == 0) and (d3 == 0)``."""

    if not isinstance(digits, torch.Tensor):
        raise TypeError(f"digits must be a torch.Tensor, got {type(digits).__name__}")
    if digits.ndim == 0 or digits.shape[-1] != 4:
        raise ValueError(f"digits must have final dimension 4, got {tuple(digits.shape)}")
    return (digits[..., 2] == 0) & (digits[..., 3] == 0)
