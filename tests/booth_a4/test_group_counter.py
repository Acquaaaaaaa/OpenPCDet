import pytest
import torch

from tools.booth_a4.group_counter import GroupCounter, GroupCounts
from tools.booth_a4.operand_mapper import OperandGroupBatch


def _batch(d2_rows, d3_rows, group_size=8):
    group_count = len(d2_rows)
    digits = torch.zeros((group_count, group_size, 4), dtype=torch.int8)
    for group, rows in enumerate(d2_rows):
        digits[group, rows, 2] = 1
    for group, rows in enumerate(d3_rows):
        digits[group, rows, 3] = -1
    logical = torch.ones((group_count, group_size), dtype=torch.bool)
    zeros = torch.zeros_like(logical)
    return OperandGroupBatch(digits, logical, zeros, zeros, zeros)


def test_group_metrics_detect_scattered_high_digits():
    counter = GroupCounter(8, window_sizes=(2,))
    counter.update(_batch([[0], [1]], [[], []]))
    counts = counter.finalize()
    metrics = counts.metrics()
    assert metrics["P_active_d2"] == pytest.approx(1.0)
    assert metrics["U_active_d2"] == pytest.approx(1 / 8)
    assert metrics["U_all_d2"] == pytest.approx(1 / 8)
    assert counts.schedule_histograms["per_group"]["mixed"] == {1: 2}
    assert counts.schedule_histograms["window_2"]["mixed"] == {2: 1}
    assert counts.schedule_histograms["layer_frame"]["mixed"] == {2: 1}


def test_physical_and_valid_density_are_distinguished():
    digits = torch.zeros((1, 8, 4), dtype=torch.int8)
    digits[0, 0, 2] = 1
    logical = torch.tensor([[True, True, False, False, False, False, False, False]])
    boundary = torch.tensor([[False, False, True, True, False, False, False, False]])
    tail = torch.tensor([[False, False, False, False, True, True, False, False]])
    pfn = torch.tensor([[False, False, False, False, False, False, True, True]])
    counter = GroupCounter(8, window_sizes=(2,))
    counter.update(OperandGroupBatch(digits, logical, boundary, tail, pfn))
    metrics = counter.finalize().metrics()
    assert metrics["U_all_d2"] == pytest.approx(1 / 8)
    assert metrics["D_valid_d2"] == pytest.approx(1 / 2)
    assert metrics["N_boundary_padding_rows"] == 2
    assert metrics["N_tail_padding_rows"] == 2
    assert metrics["N_pfn_invalid_slot_rows"] == 2


def test_counter_round_trip_and_merge_preserve_frame_schedule_boundaries():
    first_counter = GroupCounter(8, window_sizes=(2,))
    first_counter.update(_batch([[0]], [[]]))
    first = first_counter.finalize()
    second_counter = GroupCounter(8, window_sizes=(2,))
    second_counter.update(_batch([[0, 1]], [[]]))
    second = second_counter.finalize()

    merged = GroupCounts(group_size=8, window_sizes=(2,))
    merged.add(first)
    merged.add(second)
    restored = GroupCounts.from_dict(merged.to_dict())
    assert restored.schedule_histograms["layer_frame"]["mixed"] == {1: 1, 2: 1}
    assert restored.n_group_total == 2


def test_core_mode_matches_full_mode_primary_integer_counts():
    batch = _batch([[0], [1, 2], []], [[3], [], [4, 5]])
    full_counter = GroupCounter(8, window_sizes=(2, 4), retain_sensitivity=True)
    core_counter = GroupCounter(8, window_sizes=(2, 4), retain_sensitivity=False)
    full_counter.update(batch)
    core_counter.update(batch)
    full = full_counter.finalize()
    core = core_counter.finalize()

    for field_name in (
        "n_group_total",
        "n_tail_group",
        "n_tail_padding_rows",
        "n_boundary_padding_rows",
        "n_zero_insertion_rows",
        "n_pfn_invalid_slot_rows",
        "n_logical_valid_rows",
        "n_nonzero_quantized",
        "n_nonzero_quantized_logical_valid",
        "n_active_group",
        "n_nonzero_digit",
        "n_nonzero_digit_logical_valid",
        "hist_k",
        "digit_value_counts",
        "h_d2_or_d3",
        "h_d2_and_d3",
    ):
        assert getattr(core, field_name) == getattr(full, field_name)
    for series in ("mixed", "d2", "d3"):
        assert core.schedule_histograms["layer_frame"][series] == (
            full.schedule_histograms["layer_frame"][series]
        )
    assert core.window_sizes == ()
    assert not core.sensitivity_retained
    assert core.schedule_histograms["layer_frame"]["fused"] == {}


def test_batch_rejects_overlapping_row_classifications():
    digits = torch.zeros((1, 8, 4), dtype=torch.int8)
    logical = torch.ones((1, 8), dtype=torch.bool)
    with pytest.raises(ValueError, match="exactly one"):
        OperandGroupBatch(digits, logical, logical, ~logical, ~logical).validate(8)
