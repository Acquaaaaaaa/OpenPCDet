import pytest
import torch

from tools.booth_a4.booth_radix4 import (
    a4_effective_mask,
    booth_reconstruct,
    encode_modified_radix4,
)


EXPECTED_BOUNDARIES = {
    -128: [0, 0, 0, -2],
    -9: [-1, 2, -1, 0],
    -8: [0, -2, 0, 0],
    -1: [-1, 0, 0, 0],
    0: [0, 0, 0, 0],
    1: [1, 0, 0, 0],
    7: [-1, 2, 0, 0],
    8: [0, -2, 1, 0],
    127: [-1, 0, 0, 2],
}


def test_all_256_int8_codes_reconstruct_and_match_a4_oracle():
    q = torch.arange(-128, 128, dtype=torch.int16)
    digits = encode_modified_radix4(q)

    assert digits.shape == (256, 4)
    assert set(digits.unique().tolist()) <= {-2, -1, 0, 1, 2}
    assert torch.equal(booth_reconstruct(digits), q)
    assert torch.equal(a4_effective_mask(digits), (q >= -8) & (q <= 7))


@pytest.mark.parametrize(("q", "expected"), EXPECTED_BOUNDARIES.items())
def test_boundary_vectors(q, expected):
    digits = encode_modified_radix4(torch.tensor(q, dtype=torch.int8))
    assert digits.tolist() == expected


@pytest.mark.parametrize("bad", [-129, 128, 1000])
def test_rejects_values_outside_int8_domain(bad):
    with pytest.raises(ValueError, match="outside signed INT8 range"):
        encode_modified_radix4(torch.tensor([bad], dtype=torch.int16))


def test_preserves_arbitrary_input_shape():
    q = torch.tensor([[-8, -1, 0], [1, 7, 8]], dtype=torch.int8)
    assert encode_modified_radix4(q).shape == (2, 3, 4)
