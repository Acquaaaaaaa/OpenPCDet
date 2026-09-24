import pytest
import torch

from tools.booth_a4.calibration import (
    DistributionAccumulator,
    DistributionSummary,
    derive_activation_scales,
)


def test_distribution_accumulates_exact_counts_and_bounded_samples():
    accumulator = DistributionAccumulator(
        "act_test", sample_cap=5, samples_per_call=3, sample_seed=666
    )
    accumulator.update(torch.tensor([-4.0, 0.0, 2.0, 0.0]))
    accumulator.update(torch.tensor([1.0, 8.0, 0.0]))
    result = accumulator.to_dict()

    assert result["element_count"] == 7
    assert result["finite_count"] == 7
    assert result["exact_zero_count"] == 3
    assert result["float_min"] == -4.0
    assert result["float_max"] == 8.0
    assert result["max_abs"] == 8.0
    assert result["sample_count"] == 5
    assert accumulator.samples().numel() == 5


def test_nonfinite_activation_fails_loudly():
    accumulator = DistributionAccumulator("act_test")
    with pytest.raises(FloatingPointError, match="NaN=1"):
        accumulator.update(torch.tensor([0.0, float("nan")]))


def test_scale_derivation_pins_quantiles_and_denominator():
    samples = torch.arange(0, 1000, dtype=torch.float64)
    summary = DistributionSummary(
        activation_id="act_test",
        element_count=1000,
        finite_count=1000,
        exact_zero_count=1,
        float_min=0.0,
        float_max=999.0,
        max_abs=999.0,
        min_positive_abs=1.0,
        sample_count=1000,
    )
    scales, diagnostics = derive_activation_scales(summary, samples)
    assert scales["minmax"]["scale"] == pytest.approx(999 / 127)
    assert scales["p99_9"]["threshold"] == pytest.approx(998.001)
    assert diagnostics["sample_zero_count"] == 1


def test_all_zero_activation_has_null_scales():
    samples = torch.zeros(8, dtype=torch.float64)
    summary = DistributionSummary(
        activation_id="act_zero",
        element_count=8,
        finite_count=8,
        exact_zero_count=8,
        float_min=0.0,
        float_max=0.0,
        max_abs=0.0,
        sample_count=8,
    )
    scales, _ = derive_activation_scales(summary, samples)
    assert scales["all_zero_in_calibration"] is True
    assert scales["minmax"]["scale"] is None
    assert scales["p99_9"]["status"] == "uncalibratable_all_zero"


def test_sparse_nonzero_activation_reports_invalid_zero_percentile_threshold():
    samples = torch.cat((torch.zeros(1000), torch.ones(1)))
    summary = DistributionSummary(
        activation_id="act_sparse",
        element_count=1001,
        finite_count=1001,
        exact_zero_count=1000,
        float_min=0.0,
        float_max=1.0,
        max_abs=1.0,
        min_positive_abs=1.0,
        sample_count=1001,
    )
    scales, _ = derive_activation_scales(summary, samples)
    assert scales["p99_9"]["threshold"] == 0.0
    assert scales["p99_9"]["scale"] is None
    assert scales["p99_9"]["status"] == "invalid_zero_threshold"
