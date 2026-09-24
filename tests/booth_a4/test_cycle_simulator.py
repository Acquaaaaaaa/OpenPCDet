import pytest

from tools.booth_a4.cycle_simulator import CycleConfig, correction_cycles, simulate_cycles
from tools.booth_a4.group_counter import GroupCounter
from tools.booth_a4.operand_mapper import OperandGroupBatch
import torch


def _counts(high_items_per_group):
    digits = torch.zeros((len(high_items_per_group), 8, 4), dtype=torch.int8)
    for group, item_count in enumerate(high_items_per_group):
        for index in range(item_count):
            row = index % 8
            plane = 2 + index // 8
            digits[group, row, plane] = 1
    logical = torch.ones((len(high_items_per_group), 8), dtype=torch.bool)
    zeros = torch.zeros_like(logical)
    counter = GroupCounter(8, window_sizes=(2, 4))
    counter.update(OperandGroupBatch(digits, logical, zeros, zeros, zeros))
    return counter.finalize()


def test_correction_schedule_domains_apply_ceiling_at_correct_boundary():
    counts = _counts([1, 1, 1, 1])
    per_group = CycleConfig(correction_parallelism=8, correction_schedule_domain="per_group")
    window = CycleConfig(correction_parallelism=8, correction_schedule_domain="window_2")
    layer = CycleConfig(correction_parallelism=8, correction_schedule_domain="layer_frame")
    assert correction_cycles(counts, per_group)[0] == 4
    assert correction_cycles(counts, window)[0] == 2
    assert correction_cycles(counts, layer)[0] == 1


def test_vector_scalar_and_equal_resource_models_are_distinct_and_fair():
    # H=80 in one layer-frame: five groups with 16 high items each.
    counts = _counts([16, 16, 16, 16, 16])
    config = CycleConfig(
        correction_parallelism=8,
        correction_schedule_domain="layer_frame",
        main_output_parallelism=32,
        correction_output_parallelism=32,
        scalar_correction_parallelism=8,
    )
    result = simulate_cycles(counts, output_channels=128, config=config)
    assert result["C_corr_vector_tile"] == 10
    assert result["T_corr_vector"] == 4
    assert result["C_corr_vector_full"] == 40
    assert result["C_corr_scalar_full"] == 1280
    assert result["P_corr_equal_resource_scalar_ops"] == 256
    assert result["C_corr_equal_resource_scalar_full"] == 40


def test_all_zero_groups_are_safe_and_scheme_c_can_be_worse_than_b():
    counts = _counts([0, 0])
    result = simulate_cycles(counts, output_channels=16)
    assert result["C_B_tile"] == 0
    assert result["C_C_vector_tile"] == 4
    assert result["speedup_B_vs_A_tile"] is None
    assert result["speedup_C_vector_vs_B_tile"] == pytest.approx(0.0)


def test_separate_plane_queues_pay_two_ceilings():
    digits = torch.zeros((1, 8, 4), dtype=torch.int8)
    digits[0, 0, 2] = 1
    digits[0, 1, 3] = 1
    logical = torch.ones((1, 8), dtype=torch.bool)
    zeros = torch.zeros_like(logical)
    counter = GroupCounter(8, window_sizes=(2,))
    counter.update(OperandGroupBatch(digits, logical, zeros, zeros, zeros))
    counts = counter.finalize()
    mixed = CycleConfig(correction_parallelism=8, correction_plane_queue="mixed")
    separate = CycleConfig(correction_parallelism=8, correction_plane_queue="separate")
    assert correction_cycles(counts, mixed)[0] == 1
    assert correction_cycles(counts, separate)[0] == 2
