import pytest
import torch

from tools.booth_a4.counter import (
    BoothA4Accumulator,
    count_activation,
    count_activation_chunked,
)


def test_counts_and_histograms_satisfy_invariants():
    x = torch.tensor([-129.0, -8.0, -0.2, 0.0, 0.2, 7.0, 128.0])
    counts = count_activation(x, scale=1.0)
    counts.validate()

    assert counts.n_total == 7
    assert counts.n_a4 == 5
    assert counts.n_exception == 2
    assert counts.n_zero_fp32_exact == 1
    assert counts.n_zero_quantized == 3
    assert counts.n_zero_from_rounding == 2
    assert counts.n_clipped_low == 1
    assert counts.n_clipped_high == 1
    assert sum(counts.int8_hist) == counts.n_total
    assert all(sum(row) == counts.n_total for row in counts.digit_hist)
    assert counts.to_dict()["RA4_tensor"] == pytest.approx(5 / 7)


def test_merge_uses_micro_counts_not_average_of_ratios():
    first = count_activation(torch.tensor([0.0]), scale=1.0)
    second = count_activation(torch.tensor([8.0, 8.0, 8.0]), scale=1.0)

    accumulator = BoothA4Accumulator()
    accumulator.add(first)
    accumulator.add(second)
    result = accumulator.to_dict()

    assert result["n_total"] == 4
    assert result["n_a4"] == 1
    assert result["RA4_tensor"] == pytest.approx(0.25)


def test_empty_tensor_is_a_valid_counted_call():
    counts = count_activation(torch.empty(0), scale=1.0)
    counts.validate()
    assert counts.n_total == 0
    assert counts.call_count == 1
    assert counts.empty_call_count == 1
    assert counts.to_dict()["RA4_tensor"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_gpu_counts_are_identical():
    generator = torch.Generator().manual_seed(666)
    x_cpu = torch.randn(4096, generator=generator)
    cpu_counts = count_activation(x_cpu, scale=0.03125).to_dict()
    gpu_counts = count_activation(x_cpu.cuda(), scale=0.03125).to_dict()
    assert cpu_counts == gpu_counts


def test_chunked_and_unchunked_counts_are_identical_and_one_logical_call():
    generator = torch.Generator().manual_seed(666)
    x = torch.randn(10_003, generator=generator)
    direct = count_activation(x, scale=0.03125).to_dict()
    chunked = count_activation_chunked(
        x, scale=0.03125, chunk_elements=997
    ).to_dict()
    assert chunked == direct
    assert chunked["call_count"] == 1
