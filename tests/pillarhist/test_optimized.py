import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.models.backbones_3d.vfe.pillar_hist_vfe import PillarHistVFE


POINT_CLOUD_RANGE = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
VOXEL_SIZE = [0.16, 0.16, 4.0]
GRID_SIZE = [432, 496, 1]


def make_cfg(reduction_mode, coord_mode='raw_meter_xy', projection_type='linear'):
    return EasyDict({
        'NUM_BINS': 64,
        'NUM_FILTERS': [64],
        'COUNT_MODE': 'raw',
        'INTENSITY_MODE': 'mean_raw',
        'COORD_MODE': coord_mode,
        'USE_ALL_POINTS_IN_ADMITTED_PILLARS': True,
        'HISTOGRAM_DTYPE': 'float32',
        'REDUCTION_MODE': reduction_mode,
        'PROJECTION': {
            'TYPE': projection_type,
            'BIAS': projection_type == 'linear',
            'BN_EPS': 0.001,
            'BN_MOMENTUM': 0.01,
        },
    })


def make_vfe(reduction_mode, **kwargs):
    return PillarHistVFE(
        model_cfg=make_cfg(reduction_mode, **kwargs),
        num_point_features=4,
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE,
        grid_size=GRID_SIZE,
    )


def comparison_fixture(device):
    coords = torch.tensor([
        [1, 0, 495, 431],
        [0, 0, 10, 10],
        [0, 0, 0, 0],
        [1, 0, 20, 20],
    ], dtype=torch.float32, device=device)
    points = [
        [0, 0.01, -39.67, -3.0, 0.1],
        [0, 1.61, -38.07, -1.0, 0.2],
        [0, 1.62, -38.06, 0.999, 0.3],
        [1, 69.119, 39.679, 0.999, 0.4],
        [1, 3.21, -36.47, -2.5, 0.5],
        [1, 5.0, 5.0, 0.0, 0.6],
    ]
    for index in range(80):
        points.append([0, 1.61, -38.07, -2.95 + (index % 64) * 0.06, index / 80.0])
    return {
        'points': torch.tensor(points, dtype=torch.float32, device=device),
        'voxel_coords': coords,
        'batch_size': 2,
    }


def clone_batch(batch):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def test_ph_opt_yaml_builds_unique_compact_lookup_candidate(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root / 'tools')
    config = EasyDict()
    cfg_from_yaml_file('cfgs/kitti_models/pointpillar_pillarhist_opt.yaml', config)
    dataset = SimpleNamespace(
        class_names=config.CLASS_NAMES,
        point_feature_encoder=SimpleNamespace(num_point_features=4),
        grid_size=np.array(GRID_SIZE),
        point_cloud_range=np.array(POINT_CLOUD_RANGE),
        voxel_size=np.array(VOXEL_SIZE),
        depth_downsample_factor=None,
        dataset_cfg=config.DATA_CONFIG,
    )
    model = build_network(config.MODEL, num_class=3, dataset=dataset)
    assert model.vfe.reduction_mode == 'compact_lookup_segment'


@pytest.mark.parametrize('coord_mode', ['raw_meter_xy', 'normalized_xy'])
@pytest.mark.parametrize('projection_type', ['linear', 'linear_bn_relu'])
def test_tst018_cpu_histogram_concat_and_features(coord_mode, projection_type):
    torch.manual_seed(101)
    reference = make_vfe('deterministic_segment', coord_mode=coord_mode, projection_type=projection_type)
    optimized = make_vfe('compact_lookup_segment', coord_mode=coord_mode, projection_type=projection_type)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    reference.eval()
    optimized.eval()
    batch = comparison_fixture('cpu')

    ref_count, ref_intensity = reference.build_histograms(
        batch['points'], batch['voxel_coords'], batch['batch_size']
    )
    opt_count, opt_intensity = optimized.build_histograms(
        batch['points'], batch['voxel_coords'], batch['batch_size']
    )
    ref_features = reference(clone_batch(batch))['pillar_features']
    opt_features = optimized(clone_batch(batch))['pillar_features']

    assert torch.equal(opt_count, ref_count)
    torch.testing.assert_close(opt_intensity, ref_intensity, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(opt_features, ref_features, rtol=1e-5, atol=1e-6)
    assert optimized.last_diagnostics == reference.last_diagnostics


@pytest.mark.parametrize('num_voxels', [0, 1])
def test_tst018_v0_v1_contract(num_voxels):
    reference = make_vfe('deterministic_segment')
    optimized = make_vfe('compact_lookup_segment')
    optimized.load_state_dict(reference.state_dict(), strict=True)
    coords = torch.empty((0, 4), dtype=torch.float32)
    points = torch.empty((0, 5), dtype=torch.float32)
    if num_voxels:
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
    batch = {'points': points, 'voxel_coords': coords, 'batch_size': 1}
    ref = reference(clone_batch(batch))['pillar_features']
    opt = optimized(clone_batch(batch))['pillar_features']
    assert opt.shape == ref.shape == (num_voxels, 64)
    torch.testing.assert_close(opt, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='TST-018 repeatability requires CUDA')
def test_tst018_cuda_atomic_20_repeat_loss_and_gradients():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        torch.manual_seed(202)
        reference = make_vfe('deterministic_segment').cuda().train()
        optimized = make_vfe('compact_lookup_segment').cuda().train()
        optimized.load_state_dict(reference.state_dict(), strict=True)
        backend_reference = torch.nn.Linear(64, 8).cuda().train()
        backend_optimized = copy.deepcopy(backend_reference)
        batch = comparison_fixture('cuda')

        def backward(vfe, backend):
            vfe.zero_grad(set_to_none=True)
            backend.zero_grad(set_to_none=True)
            features = vfe(clone_batch(batch))['pillar_features']
            loss = backend(features).square().mean()
            loss.backward()
            return (
                features.detach().clone(),
                loss.detach().clone(),
                vfe.projection.weight.grad.detach().clone(),
                backend.weight.grad.detach().clone(),
            )

        expected = backward(reference, backend_reference)
        repeated = []
        for _ in range(20):
            actual = backward(optimized, backend_optimized)
            torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-6)
            for actual_grad, expected_grad in zip(actual[2:], expected[2:]):
                torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)
                if actual_grad.norm() > 1e-8 and expected_grad.norm() > 1e-8:
                    cosine = torch.nn.functional.cosine_similarity(
                        actual_grad.flatten(), expected_grad.flatten(), dim=0
                    )
                    assert float(cosine) >= 0.9999
            repeated.append(actual)
        for index in range(1, len(repeated)):
            torch.testing.assert_close(repeated[index][0], repeated[0][0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(repeated[index][1], repeated[0][1], rtol=1e-5, atol=1e-6)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='TST-018 detector comparison requires CUDA')
def test_tst018_full_detector_loss_and_backend_gradient(monkeypatch):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        repo_root = Path(__file__).resolve().parents[2]
        monkeypatch.chdir(repo_root / 'tools')
        config = EasyDict()
        cfg_from_yaml_file('cfgs/kitti_models/pointpillar_pillarhist.yaml', config)
        dataset = SimpleNamespace(
            class_names=config.CLASS_NAMES,
            point_feature_encoder=SimpleNamespace(num_point_features=4),
            grid_size=np.array(GRID_SIZE),
            point_cloud_range=np.array(POINT_CLOUD_RANGE),
            voxel_size=np.array(VOXEL_SIZE),
            depth_downsample_factor=None,
            dataset_cfg=config.DATA_CONFIG,
        )
        torch.manual_seed(303)
        reference = build_network(config.MODEL, num_class=3, dataset=dataset).cuda().train()
        opt_cfg = copy.deepcopy(config.MODEL)
        opt_cfg.VFE.REDUCTION_MODE = 'compact_lookup_segment'
        optimized = build_network(opt_cfg, num_class=3, dataset=dataset).cuda().train()
        optimized.load_state_dict(reference.state_dict(), strict=True)
        coords = torch.tensor([
            [0, 0, 240, 30], [0, 0, 241, 30],
            [0, 0, 240, 31], [0, 0, 241, 31],
        ], dtype=torch.int32, device='cuda')
        points = torch.tensor([
            [0, 4.81, -1.27, -1.20, 0.20],
            [0, 4.82, -1.11, -1.00, 0.40],
            [0, 4.97, -1.27, -0.80, 0.60],
            [0, 4.98, -1.11, -0.60, 0.80],
        ], dtype=torch.float32, device='cuda')
        gt_boxes = torch.tensor([[[
            4.90, -1.20, -1.00, 3.90, 1.60, 1.56, 0.0, 1.0
        ]]], dtype=torch.float32, device='cuda')

        def backward(model):
            model.zero_grad(set_to_none=True)
            result, _, _ = model({
                'points': points.clone(), 'voxel_coords': coords.clone(),
                'batch_size': 1, 'gt_boxes': gt_boxes.clone(),
            })
            loss = result['loss']
            loss.backward()
            return (
                loss.detach(),
                model.vfe.projection.weight.grad.detach().clone(),
                model.dense_head.conv_cls.weight.grad.detach().clone(),
            )

        expected = backward(reference)
        actual = backward(optimized)
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(actual[2], expected[2], rtol=1e-4, atol=1e-6)
    finally:
        torch.use_deterministic_algorithms(previous)
