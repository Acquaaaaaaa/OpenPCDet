import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.models.backbones_3d import vfe as vfe_registry
from pcdet.models.backbones_3d.vfe.pillar_hist_vfe import PillarHistVFE


POINT_CLOUD_RANGE = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
VOXEL_SIZE = [0.16, 0.16, 4.0]
GRID_SIZE = [432, 496, 1]


def make_cfg(coord_mode='raw_meter_xy', projection_type='linear'):
    return EasyDict({
        'NUM_BINS': 64,
        'NUM_FILTERS': [64],
        'COUNT_MODE': 'raw',
        'INTENSITY_MODE': 'mean_raw',
        'COORD_MODE': coord_mode,
        'USE_ALL_POINTS_IN_ADMITTED_PILLARS': True,
        'HISTOGRAM_DTYPE': 'float32',
        'REDUCTION_MODE': 'deterministic_segment',
        'PROJECTION': {
            'TYPE': projection_type,
            'BIAS': projection_type == 'linear',
            'BN_EPS': 0.001,
            'BN_MOMENTUM': 0.01,
        },
    })


def make_vfe(**kwargs):
    return PillarHistVFE(
        model_cfg=make_cfg(**kwargs),
        num_point_features=4,
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE,
        grid_size=GRID_SIZE,
    )


def sample_batch():
    return {
        'points': torch.tensor([
            [0, 0.01, -39.67, -2.99, 0.1],
            [0, 0.02, -39.66, -2.98, 0.2],
        ], dtype=torch.float32),
        'voxel_coords': torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
        'batch_size': 1,
    }


def build_test_network(monkeypatch):
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
    return build_network(config.MODEL, num_class=3, dataset=dataset)


def test_registry_and_both_projection_coordinate_variants_construct():
    assert vfe_registry.__all__['PillarHistVFE'] is PillarHistVFE
    paper_literal = make_vfe()
    engineering = make_vfe(
        coord_mode='normalized_xy', projection_type='linear_bn_relu'
    )

    assert paper_literal.get_output_feature_dim() == 64
    assert isinstance(paper_literal.projection, torch.nn.Linear)
    assert isinstance(engineering.projection, torch.nn.Sequential)
    normalized_center = engineering._pillar_centers(
        torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    )
    assert normalized_center.shape == (1, 2)
    assert torch.all((normalized_center >= -1) & (normalized_center <= 1))


def test_outer_autocast_keeps_reference_and_projection_float32():
    devices = [('cpu', torch.bfloat16)]
    if torch.cuda.is_available():
        devices.append(('cuda', torch.float16))

    for device, autocast_dtype in devices:
        model = make_vfe().to(device)
        batch = sample_batch()
        batch['points'] = batch['points'].to(device)
        batch['voxel_coords'] = batch['voxel_coords'].to(device)
        captured = {}

        def capture_input(_module, args):
            captured['dtype'] = args[0].dtype

        handle = model.projection.register_forward_pre_hook(capture_input)
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            output = model(batch)['pillar_features']
        handle.remove()

        assert captured['dtype'] == torch.float32
        diagnostics = model.last_diagnostics
        assert diagnostics['count_hist_storage_dtype'] == 'torch.int32'
        assert diagnostics['count_feature_dtype'] == 'torch.float32'
        assert diagnostics['intensity_mean_dtype'] == 'torch.float32'
        assert diagnostics['center_dtype'] == 'torch.float32'
        assert diagnostics['histogram_feature_dtype'] == 'torch.float32'
        assert diagnostics['pillar_feature_dtype'] == 'torch.float32'
        assert output.dtype == torch.float32


def test_projection_backward_and_checkpoint_round_trip():
    torch.manual_seed(7)
    model = make_vfe()
    fixed_batch = sample_batch()
    expected = model(fixed_batch)['pillar_features'].detach().clone()
    loss = model(sample_batch())['pillar_features'].square().mean()
    loss.backward()

    assert model.projection.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert model.projection.bias.grad is not None
    assert torch.isfinite(model.projection.bias.grad).all()

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = make_vfe()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = restored(sample_batch())['pillar_features']
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_yaml_builds_complete_pointpillar_with_unchanged_head_channels(monkeypatch):
    model = build_test_network(monkeypatch)

    assert isinstance(model.vfe, PillarHistVFE)
    assert model.map_to_bev_module.num_bev_features == 64
    assert model.dense_head.conv_cls.out_channels == 18
    assert model.dense_head.conv_box.out_channels == 42
    assert model.dense_head.conv_dir_cls.out_channels == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason='full detector smoke requires CUDA')
def test_full_detector_loss_backward_reaches_projection(monkeypatch):
    model = build_test_network(monkeypatch).cuda().train()
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

    ret_dict, tensorboard, _ = model({
        'points': points,
        'voxel_coords': coords,
        'batch_size': 1,
        'gt_boxes': gt_boxes,
    })
    loss = ret_dict['loss']

    assert torch.isfinite(loss)
    head_state = model.dense_head.forward_ret_dict
    assert head_state['cls_preds'].shape == (1, 248, 216, 18)
    assert head_state['box_preds'].shape == (1, 248, 216, 42)
    assert head_state['dir_cls_preds'].shape == (1, 248, 216, 12)
    assert head_state['box_cls_labels'].shape == (1, 248 * 216 * 6)
    assert int((head_state['box_cls_labels'] > 0).sum()) > 0
    assert all(np.isfinite(value) for value in tensorboard.values())
    loss.backward()
    projection_grad = model.vfe.projection.weight.grad
    assert projection_grad is not None
    assert torch.isfinite(projection_grad).all()
    assert torch.count_nonzero(projection_grad) > 0
    backend_grad = model.dense_head.conv_cls.weight.grad
    assert backend_grad is not None
    assert torch.isfinite(backend_grad).all()
    assert torch.count_nonzero(backend_grad) > 0
