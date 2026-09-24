import pytest
import torch
import torch.nn as nn

from tools.booth_a4.booth_radix4 import encode_modified_radix4
from tools.booth_a4.operand_mapper import (
    iter_conv2d_groups,
    iter_conv_transpose2d_groups,
    iter_conv_transpose2d_zero_insertion_groups,
    iter_linear_groups,
    iter_logical_contiguous_groups,
)


def _merge(batches, field):
    return torch.cat([getattr(batch, field) for batch in batches])


def test_linear_preserves_token_boundaries_and_classifies_fixed_slots():
    module = nn.Linear(10, 4, bias=False)
    q = torch.tensor(
        [[[1] * 10, [7] * 10]], dtype=torch.int8
    )
    digits = encode_modified_radix4(q)
    batches = list(
        iter_linear_groups(
            digits,
            module,
            group_size=8,
            valid_token_mask=torch.tensor([[True, False]]),
        )
    )
    mapped = _merge(batches, "digits")
    assert mapped.shape == (4, 8, 4)
    assert int(_merge(batches, "logical_valid").sum()) == 10
    assert int(_merge(batches, "pfn_invalid_slot").sum()) == 10
    assert int(_merge(batches, "tail_padding").sum()) == 12
    assert not bool(_merge(batches, "boundary_padding").any())


def test_linear_valid_points_only_drops_invalid_token_streams():
    module = nn.Linear(10, 4, bias=False)
    digits = encode_modified_radix4(torch.ones((1, 2, 10), dtype=torch.int8))
    batches = list(
        iter_linear_groups(
            digits,
            module,
            group_size=8,
            token_policy="valid_points_only",
            valid_token_mask=torch.tensor([[True, False]]),
        )
    )
    assert _merge(batches, "digits").shape == (2, 8, 4)
    assert not bool(_merge(batches, "pfn_invalid_slot").any())


def test_conv2d_physical_padding_and_tail_are_separate():
    module = nn.Conv2d(1, 2, 3, padding=1, bias=False)
    digits = encode_modified_radix4(torch.tensor([[[[8]]]], dtype=torch.int8))
    batches = list(iter_conv2d_groups(digits, module, group_size=8))
    assert _merge(batches, "digits").shape == (2, 8, 4)
    assert int(_merge(batches, "logical_valid").sum()) == 1
    assert int(_merge(batches, "boundary_padding").sum()) == 8
    assert int(_merge(batches, "tail_padding").sum()) == 7
    mapped = _merge(batches, "digits")
    assert int((mapped[:, :, 2] != 0).sum()) == 1


def test_conv2d_ideal_padding_compacts_valid_operands():
    module = nn.Conv2d(1, 2, 3, padding=1, bias=False)
    digits = encode_modified_radix4(torch.tensor([[[[8]]]], dtype=torch.int8))
    batches = list(
        iter_conv2d_groups(
            digits, module, group_size=8, padding_policy="ideal_padding_gated"
        )
    )
    assert _merge(batches, "digits").shape == (1, 8, 4)
    assert int(_merge(batches, "logical_valid").sum()) == 1
    assert not bool(_merge(batches, "boundary_padding").any())
    assert int(_merge(batches, "tail_padding").sum()) == 7


def test_conv2d_explicit_external_padding_matches_zero_pad_capture():
    module = nn.Conv2d(1, 1, 3, stride=2, padding=0, bias=False)
    digits = encode_modified_radix4(torch.ones((1, 1, 4, 4), dtype=torch.int8))
    batches = list(
        iter_conv2d_groups(digits, module, group_size=8, effective_padding=1)
    )
    # output is 2x2, K=9, hence two groups per output position
    assert _merge(batches, "digits").shape[0] == 8


@pytest.mark.parametrize("group_size", [4, 8, 16, 32])
def test_conv2d_group_count_formula_across_group_sizes(group_size):
    module = nn.Conv2d(3, 5, 3, padding=1, bias=False)
    digits = encode_modified_radix4(torch.ones((1, 3, 4, 5), dtype=torch.int8))
    batches = list(iter_conv2d_groups(digits, module, group_size=group_size))
    expected = 4 * 5 * ((3 * 3 * 3 + group_size - 1) // group_size)
    assert _merge(batches, "digits").shape[0] == expected


def test_conv2d_chunk_size_does_not_change_group_sequence():
    module = nn.Conv2d(3, 5, 3, padding=1, bias=False)
    generator = torch.Generator().manual_seed(666)
    q = torch.randint(-128, 128, (1, 3, 4, 5), generator=generator, dtype=torch.int16)
    digits = encode_modified_radix4(q)
    small = list(iter_conv2d_groups(digits, module, group_size=8, chunk_streams=2))
    large = list(iter_conv2d_groups(digits, module, group_size=8, chunk_streams=64))
    for field in (
        "digits", "logical_valid", "boundary_padding", "tail_padding", "pfn_invalid_slot"
    ):
        assert torch.equal(_merge(small, field), _merge(large, field))


def test_conv_transpose_direct_scatter_replays_each_kernel_offset():
    module = nn.ConvTranspose2d(3, 5, 2, stride=2, bias=False)
    digits = encode_modified_radix4(torch.ones((1, 3, 2, 2), dtype=torch.int8))
    batches = list(iter_conv_transpose2d_groups(digits, module, group_size=8))
    # B*H*W*Kh*Kw streams, one group per Cin=3 stream.
    assert _merge(batches, "digits").shape == (16, 8, 4)
    assert int(_merge(batches, "logical_valid").sum()) == 16 * 3
    assert int(_merge(batches, "tail_padding").sum()) == 16 * 5


def test_conv_transpose_zero_insertion_is_counted_separately():
    module = nn.ConvTranspose2d(1, 2, 2, stride=2, bias=False)
    digits = encode_modified_radix4(torch.ones((1, 1, 2, 2), dtype=torch.int8))
    batches = list(
        iter_conv_transpose2d_zero_insertion_groups(digits, module, group_size=8)
    )
    assert _merge(batches, "digits").shape == (16, 8, 4)
    assert int(_merge(batches, "logical_valid").sum()) == 16
    assert int(_merge(batches, "zero_insertion_mask").sum()) > 0
    classified = sum(
        _merge(batches, name).to(torch.int8)
        for name in (
            "logical_valid", "boundary_padding", "tail_padding", "pfn_invalid_slot",
            "zero_insertion_mask",
        )
    )
    assert bool(torch.all(classified == 1))


def test_logical_contiguous_baseline_only_pads_the_final_global_group():
    digits = encode_modified_radix4(torch.ones((2, 3), dtype=torch.int8))
    batches = list(iter_logical_contiguous_groups(digits, group_size=4, chunk_streams=1))
    assert _merge(batches, "digits").shape == (2, 4, 4)
    assert int(_merge(batches, "logical_valid").sum()) == 6
    assert int(_merge(batches, "tail_padding").sum()) == 2
