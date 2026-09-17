import pytest
import torch
from easydict import EasyDict

from pcdet.models.backbones_3d.vfe.pillar_hist_vfe import PillarHistVFE


def make_vfe(**overrides):
    cfg = EasyDict({
        'NUM_BINS': 64,
        'NUM_FILTERS': [64],
        'COUNT_MODE': 'raw',
        'INTENSITY_MODE': 'mean_raw',
        'COORD_MODE': 'raw_meter_xy',
        'USE_ALL_POINTS_IN_ADMITTED_PILLARS': True,
        'HISTOGRAM_DTYPE': 'float32',
        'REDUCTION_MODE': 'deterministic_segment',
        'PROJECTION': {'TYPE': 'linear', 'BIAS': True},
    })
    cfg.update(overrides)
    return PillarHistVFE(
        model_cfg=cfg,
        num_point_features=4,
        voxel_size=[0.16, 0.16, 4.0],
        point_cloud_range=[0.0, -39.68, -3.0, 69.12, 39.68, 1.0],
        grid_size=[432, 496, 1],
    )


def test_z_and_xy_half_open_boundaries_and_empty_bins():
    vfe = make_vfe()
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    below_first_boundary = torch.nextafter(
        torch.tensor(-2.9375), torch.tensor(float('-inf'))
    ).item()
    below_z_max = torch.nextafter(
        torch.tensor(1.0), torch.tensor(float('-inf'))
    ).item()
    points = torch.tensor([
        [0, 0.00, -39.68, -3.0000, 0.1],
        [0, 0.01, -39.67, below_first_boundary, 0.2],
        [0, 0.01, -39.67, -2.9375, 0.3],
        [0, 0.01, -39.67, below_z_max, 0.4],
        [0, 0.01, -39.67, 1.0000, 0.5],
        [0, 69.12, -39.67, -2.9900, 0.6],
        [0, 0.01, 39.68, -2.9900, 0.7],
    ], dtype=torch.float32)

    count, intensity = vfe.build_histograms(points, coords, batch_size=1)

    assert int(count.sum()) == 4
    assert int(count[0, 0]) == 2
    assert int(count[0, 1]) == 1
    assert int(count[0, 63]) == 1
    assert torch.all(intensity[count == 0] == 0)
    assert torch.isfinite(intensity).all()
    assert vfe.last_diagnostics['protective_clamp_count'] == 0


@pytest.mark.parametrize(
    'coords',
    [
        [[1, 0, 0, 0]],
        [[0, 1, 0, 0]],
        [[0, 0, 496, 0]],
        [[0, 0, 0, 432]],
        [[-1, 0, 0, 0]],
        [[0, 0, -1, 0]],
        [[0, 0, 0, -1]],
    ],
)
def test_invalid_active_coordinate_ranges_raise(coords):
    vfe = make_vfe()
    with pytest.raises(ValueError):
        vfe.build_histograms(
            torch.empty((0, 5), dtype=torch.float32),
            torch.tensor(coords, dtype=torch.float32),
            batch_size=1,
        )


def test_duplicate_and_non_integer_discrete_inputs_raise():
    vfe = make_vfe()
    empty_points = torch.empty((0, 5), dtype=torch.float32)
    with pytest.raises(ValueError, match='duplicate'):
        vfe.build_histograms(
            empty_points,
            torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int32),
            batch_size=1,
        )
    with pytest.raises(ValueError, match='integer-valued'):
        vfe.build_histograms(
            empty_points,
            torch.tensor([[0, 0, 0.5, 0]], dtype=torch.float32),
            batch_size=1,
        )
    with pytest.raises(ValueError, match='integer-valued'):
        vfe.build_histograms(
            torch.tensor([[0.5, 0.01, -39.67, -2.99, 0.1]]),
            torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
            batch_size=1,
        )

    for invalid_batch in (-1.0, 1.0):
        with pytest.raises(ValueError, match='batch ids'):
            vfe.build_histograms(
                torch.tensor([[invalid_batch, 0.01, -39.67, -2.99, 0.1]]),
                torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
                batch_size=1,
            )


def test_integer_valued_float32_discrete_inputs_are_accepted():
    vfe = make_vfe()
    count, _ = vfe.build_histograms(
        torch.tensor([[0.0, 0.01, -39.67, -2.99, 0.1]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        batch_size=1,
    )
    assert int(count.sum()) == 1


def test_maximum_legal_packed_keys_are_distinct_and_admitted():
    vfe = make_vfe()
    coords = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 495, 431],
        [1, 0, 0, 0],
        [1, 0, 495, 431],
    ], dtype=torch.int32)
    keys = coords[:, 0].long() * (496 * 432) + coords[:, 2].long() * 432 + coords[:, 3].long()
    assert torch.unique(keys).numel() == coords.shape[0]
    assert int(keys[-1]) == 2 * 496 * 432 - 1
    below_z_max = torch.nextafter(
        torch.tensor(1.0), torch.tensor(float('-inf'))
    ).item()
    points = torch.tensor([
        [0, 0.0, -39.68, -3.0, 0.1],
        [0, 69.11, 39.67, below_z_max, 0.2],
        [1, 0.0, -39.68, -3.0, 0.3],
        [1, 69.11, 39.67, below_z_max, 0.4],
    ], dtype=torch.float32)
    count, _ = vfe.build_histograms(points, coords, batch_size=2)
    assert torch.equal(count.sum(dim=1), torch.ones(4, dtype=torch.int64))


def test_nonzero_protective_clamp_is_never_silent():
    vfe = make_vfe()
    vfe.bin_height = vfe.bin_height / 2
    with pytest.raises(RuntimeError, match='protective height-bin clamping'):
        vfe.build_histograms(
            torch.tensor([[0, 0.01, -39.67, 0.9, 0.1]], dtype=torch.float32),
            torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
            batch_size=1,
        )


def test_v_one_and_v_zero_shapes_are_preserved():
    vfe = make_vfe()
    single = vfe({
        'points': torch.tensor([[0, 0.01, -39.67, -2.99, 0.1]]),
        'voxel_coords': torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
        'batch_size': 1,
    })['pillar_features']
    empty = vfe({
        'points': torch.empty((0, 5), dtype=torch.float32),
        'voxel_coords': torch.empty((0, 4), dtype=torch.int32),
        'batch_size': 1,
    })['pillar_features']

    assert single.shape == (1, 64)
    assert empty.shape == (0, 64)
    assert single.dtype == torch.float32
    assert empty.dtype == torch.float32


@pytest.mark.parametrize(
    'field,value',
    [
        ('COUNT_MODE', 'normalized'),
        ('INTENSITY_MODE', 'sum'),
        ('COORD_MODE', 'unknown'),
        ('HISTOGRAM_DTYPE', 'float16'),
        ('REDUCTION_MODE', 'atomic'),
        ('USE_ALL_POINTS_IN_ADMITTED_PILLARS', False),
    ],
)
def test_unknown_or_disallowed_modes_raise(field, value):
    with pytest.raises(ValueError):
        make_vfe(**{field: value})
