import pytest
import torch

from tools.booth_a4.quantizer import quantize_symmetric_int8, scale_from_threshold


def test_nearest_even_rounding_for_positive_and_negative_ties():
    x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    result = quantize_symmetric_int8(x, scale=1.0)
    assert result.q.tolist() == [-2, -2, 0, 0, 2, 2]


def test_clamp_and_clipping_are_derived_from_preclamp_codes():
    x = torch.tensor([-129.0, -128.0, 127.0, 128.0])
    result = quantize_symmetric_int8(x, scale=1.0)
    assert result.q.tolist() == [-128, -128, 127, 127]
    assert result.clipped_low.tolist() == [True, False, False, False]
    assert result.clipped_high.tolist() == [False, False, False, True]


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale must be finite and positive"):
        quantize_symmetric_int8(torch.tensor([0.0]), scale)


def test_rejects_nan_or_inf_activations():
    with pytest.raises(ValueError, match="NaN or Inf"):
        quantize_symmetric_int8(torch.tensor([0.0, float("nan")]), 1.0)


def test_scale_from_threshold_uses_127_denominator():
    assert scale_from_threshold(12.7) == pytest.approx(0.1)
