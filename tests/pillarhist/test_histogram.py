import pytest
import torch
from easydict import EasyDict

from pcdet.models.backbones_3d.vfe.pillar_hist_vfe import (
    PillarHistVFE,
    pillar_histogram_cpu_oracle,
)


POINT_CLOUD_RANGE = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
VOXEL_SIZE = [0.16, 0.16, 4.0]
GRID_SIZE = [432, 496, 1]


def make_vfe():
    return PillarHistVFE(
        model_cfg=EasyDict({
            'NUM_BINS': 64,
            'NUM_FILTERS': [64],
            'COUNT_MODE': 'raw',
            'INTENSITY_MODE': 'mean_raw',
            'COORD_MODE': 'raw_meter_xy',
            'USE_ALL_POINTS_IN_ADMITTED_PILLARS': True,
            'HISTOGRAM_DTYPE': 'float32',
            'REDUCTION_MODE': 'deterministic_segment',
            'PROJECTION': {'TYPE': 'linear', 'BIAS': True},
        }),
        num_point_features=4,
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE,
        grid_size=GRID_SIZE,
    )


def test_cpu_oracle_and_reference_point_conservation():
    vfe = make_vfe()
    coords = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 2, 3],
        [1, 0, 0, 0],
    ], dtype=torch.int32)
    points = torch.tensor([
        [0, 0.01, -39.67, -2.99, 0.10],
        [0, 0.02, -39.66, -2.98, 0.30],
        [0, 0.49, -39.35, -1.00, 0.50],
        [1, 0.01, -39.67, -2.99, 0.70],
        [0, 10.00, 0.00, 0.00, 0.90],
    ], dtype=torch.float32)

    count, intensity = vfe.build_histograms(points, coords, batch_size=2)
    oracle_count, oracle_intensity = pillar_histogram_cpu_oracle(
        points,
        coords,
        batch_size=2,
        num_bins=64,
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE,
        grid_size=GRID_SIZE,
    )

    assert count.dtype == torch.int32
    assert intensity.dtype == torch.float32
    assert torch.equal(count, oracle_count)
    torch.testing.assert_close(intensity, oracle_intensity, rtol=1e-5, atol=1e-6)
    assert int(count.sum()) == 4
    assert vfe.last_diagnostics['num_admitted_points'] == 4
    assert vfe.last_diagnostics['protective_clamp_count'] == 0


def test_all_points_are_used_beyond_voxelizer_max32():
    vfe = make_vfe()
    coords = torch.tensor([[0, 0, 10, 10]], dtype=torch.int32)
    points = torch.zeros((40, 5), dtype=torch.float32)
    points[:, 1] = 10 * 0.16 + 0.01
    points[:, 2] = -39.68 + 10 * 0.16 + 0.01
    points[:, 3] = -2.99
    points[:, 4] = torch.arange(40, dtype=torch.float32) / 40

    count, intensity = vfe.build_histograms(points, coords, batch_size=1)

    assert int(count.sum()) == 40
    assert int(count[0, 0]) == 40
    torch.testing.assert_close(intensity[0, 0], points[:, 4].mean())
    assert int(count[0].sum()) != min(points.shape[0], 32)


def test_admission_cap_counts_all_points_only_in_admitted_pillars():
    vfe = make_vfe()
    coords = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=torch.int32)
    points = []
    for cx, num_points in ((0, 40), (1, 41), (2, 42)):
        for point_idx in range(num_points):
            points.append([
                0,
                cx * 0.16 + 0.01,
                -39.67,
                -2.99,
                point_idx / 100.0,
            ])
    points = torch.tensor(points, dtype=torch.float32)

    count, _ = vfe.build_histograms(points, coords, batch_size=1)

    assert int(count.sum()) == 81
    assert int(count[0].sum()) == 40
    assert int(count[1].sum()) == 41
    assert vfe.last_diagnostics['overflow_pillars'] == 1
    assert vfe.last_diagnostics['overflow_points'] == 42


def test_row_alignment_batch_isolation_and_point_order_invariance():
    vfe = make_vfe()
    coords = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    ], dtype=torch.int32)
    points = torch.tensor([
        [0, 0.01, -39.67, -2.99, 0.10],
        [1, 0.01, -39.67, -2.99, 0.90],
        [0, 0.17, -39.51, -2.99, 0.50],
        [0, 0.02, -39.66, -2.98, 0.30],
    ], dtype=torch.float32)

    count_a, intensity_a = vfe.build_histograms(points, coords, batch_size=2)
    count_b, intensity_b = vfe.build_histograms(
        points[torch.tensor([2, 0, 3, 1])], coords, batch_size=2
    )

    assert int(count_a[0].sum()) == 1
    assert int(count_a[1].sum()) == 1
    assert int(count_a[2].sum()) == 2
    assert intensity_a[0, 0].item() == torch.tensor(0.9).item()
    assert torch.equal(count_a, count_b)
    torch.testing.assert_close(intensity_a, intensity_b, rtol=1e-5, atol=1e-6)


def test_empty_middle_batch_does_not_cross_contaminate_neighbors():
    vfe = make_vfe()
    coords = torch.tensor([
        [0, 0, 0, 0],
        [2, 0, 0, 0],
    ], dtype=torch.int32)
    points = torch.tensor([
        [0, 0.01, -39.67, -2.99, 0.1],
        [1, 0.01, -39.67, -2.99, 0.5],
        [2, 0.01, -39.67, -2.99, 0.9],
    ], dtype=torch.float32)

    count, intensity = vfe.build_histograms(points, coords, batch_size=3)

    assert count.shape == (2, 64)
    assert int(count[0].sum()) == 1
    assert int(count[1].sum()) == 1
    assert intensity[0, 0].item() == torch.tensor(0.1).item()
    assert intensity[1, 0].item() == torch.tensor(0.9).item()
    assert vfe.last_diagnostics['overflow_pillars'] == 1
    assert vfe.last_diagnostics['overflow_points'] == 1


def test_reference_is_deterministic():
    vfe = make_vfe()
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    points = torch.tensor([
        [0, 0.01, -39.67, -2.99, 0.1],
        [0, 0.02, -39.66, -2.98, 0.2],
    ], dtype=torch.float32)

    first = vfe({'points': points, 'voxel_coords': coords, 'batch_size': 1})[
        'pillar_features'
    ]
    second = vfe({'points': points, 'voxel_coords': coords, 'batch_size': 1})[
        'pillar_features'
    ]

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='TST-015/016 require CUDA')
def test_cpu_cuda_reference_and_cuda_determinism():
    torch.manual_seed(11)
    cpu_vfe = make_vfe()
    cuda_vfe = make_vfe().cuda()
    cuda_vfe.load_state_dict(cpu_vfe.state_dict())
    coords = torch.tensor([
        [1, 0, 495, 431],
        [0, 0, 0, 0],
        [0, 0, 20, 10],
    ], dtype=torch.int32)
    points = torch.tensor([
        [0, 0.01, -39.67, -3.0, 0.1],
        [0, 1.61, -36.47, -1.0, 0.2],
        [1, 69.11, 39.67, torch.nextafter(torch.tensor(1.0), torch.tensor(float('-inf'))), 0.3],
        [1, 69.10, 39.66, 0.5, 0.4],
    ], dtype=torch.float32)

    cpu_count, cpu_intensity = cpu_vfe.build_histograms(points, coords, batch_size=2)
    cuda_count, cuda_intensity = cuda_vfe.build_histograms(
        points.cuda(), coords.cuda(), batch_size=2
    )

    assert torch.equal(cuda_count.cpu(), cpu_count)
    torch.testing.assert_close(cuda_intensity.cpu(), cpu_intensity, rtol=1e-5, atol=1e-6)
    batch = {'points': points.cuda(), 'voxel_coords': coords.cuda(), 'batch_size': 2}
    first = cuda_vfe(dict(batch))['pillar_features']
    second = cuda_vfe(dict(batch))['pillar_features']
    torch.testing.assert_close(first, second, rtol=0, atol=0)
