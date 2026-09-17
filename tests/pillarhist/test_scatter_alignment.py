import pytest
import torch
from easydict import EasyDict

from pcdet.models.backbones_2d.map_to_bev.pointpillar_scatter import (
    PointPillarScatter,
)


def make_scatter():
    return PointPillarScatter(
        model_cfg=EasyDict({'NUM_BEV_FEATURES': 1}),
        grid_size=[2, 2, 1],
    )


def scatter(features, coords):
    return make_scatter()({
        'pillar_features': features,
        'voxel_coords': coords,
        'batch_size': 1,
    })['spatial_features']


def test_two_by_two_spatial_alignment_and_joint_permutation_invariance():
    coords = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
    ], dtype=torch.int32)
    features = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    expected = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    original = scatter(features, coords)
    permutation = torch.tensor([2, 0, 3, 1])
    jointly_permuted = scatter(features[permutation], coords[permutation])
    feature_only_permuted = scatter(features[permutation], coords)

    assert torch.equal(original, expected)
    assert torch.equal(jointly_permuted, expected)
    assert not torch.equal(feature_only_permuted, expected)


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason='TST-014B: fixed upstream scatter calls max() for an empty coordinate set',
)
def test_upstream_scatter_global_empty_batch_limitation():
    scatter(
        torch.empty((0, 1), dtype=torch.float32),
        torch.empty((0, 4), dtype=torch.int32),
    )
