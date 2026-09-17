#!/usr/bin/env python3
"""Formal R6 reference profiler for the frozen PillarHist v1.3 entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stats(values):
    return {
        'n': len(values),
        'min_ms': min(values),
        'p50_ms': percentile(values, 0.5),
        'p95_ms': percentile(values, 0.95),
        'max_ms': max(values),
        'mean_ms': statistics.fmean(values),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def make_batch(dataset, frame_ids):
    by_id = {
        str(info['point_cloud']['lidar_idx']): index
        for index, info in enumerate(dataset.kitti_infos)
    }
    missing = [frame_id for frame_id in frame_ids if frame_id not in by_id]
    if missing:
        raise KeyError(f'fixture frame IDs not found: {missing}')
    batch = dataset.collate_batch([dataset[by_id[frame_id]] for frame_id in frame_ids])
    actual = [str(value) for value in np.asarray(batch['frame_id']).reshape(-1)]
    if actual != frame_ids:
        raise AssertionError(f'fixture mismatch: expected {frame_ids}, got {actual}')
    return batch


def clone_batch(batch):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def measure_cuda(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return {
        'latency': stats(values),
        'samples_ms': values,
        'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
        'peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
    }


@torch.no_grad()
def component_pass(vfe, batch):
    points = batch['points']
    voxel_coords = batch['voxel_coords']
    batch_size = int(batch['batch_size'])
    marks = [torch.cuda.Event(enable_timing=True) for _ in range(6)]

    marks[0].record()
    coords, point_batch, active_key = vfe._validate_inputs(points, voxel_coords, batch_size)
    num_voxels = coords.shape[0]
    points_fp32 = points.float()
    xyz = points_fp32[:, 1:4]
    in_roi = (
        (xyz[:, 0] >= vfe.x_min) & (xyz[:, 0] < vfe.x_max)
        & (xyz[:, 1] >= vfe.y_min) & (xyz[:, 1] < vfe.y_max)
        & (xyz[:, 2] >= vfe.z_min) & (xyz[:, 2] < vfe.z_max)
    )
    roi_points = points_fp32[in_roi]
    roi_batch = point_batch[in_roi]
    geometry = points_fp32.new_tensor([
        vfe.x_min, vfe.y_min, vfe.z_min, vfe.voxel_x, vfe.voxel_y, vfe.bin_height
    ]).double()
    x_min, y_min, z_min, voxel_x, voxel_y, bin_height = geometry.unbind()
    cx = torch.floor((roi_points[:, 1].double() - x_min) / voxel_x).long()
    cy = torch.floor((roi_points[:, 2].double() - y_min) / voxel_y).long()
    bins = torch.floor((roi_points[:, 3].double() - z_min) / bin_height).long().clamp(0, vfe.num_bins - 1)
    point_key = roi_batch * (vfe.ny * vfe.nx) + cy * vfe.nx + cx
    sorted_key, permutation = torch.sort(active_key, stable=True)
    positions = torch.searchsorted(sorted_key, point_key)
    safe_positions = positions.clamp(max=num_voxels - 1)
    admitted = (positions < num_voxels) & (sorted_key[safe_positions] == point_key)
    rows = permutation[safe_positions[admitted]]
    admitted_bins = bins[admitted]
    admitted_intensity = roi_points[admitted, 4].float()
    segment_key = rows * vfe.num_bins + admitted_bins
    marks[1].record()

    sorted_segment, segment_permutation = torch.sort(segment_key, stable=True)
    sorted_intensity = admitted_intensity[segment_permutation]
    unique_segment, segment_counts = torch.unique_consecutive(sorted_segment, return_counts=True)
    count_hist = torch.zeros((num_voxels, vfe.num_bins), dtype=torch.int32, device=points.device)
    count_hist.view(-1)[unique_segment] = segment_counts.to(torch.int32)
    marks[2].record()

    segment_sums = torch.segment_reduce(sorted_intensity, reduce='sum', lengths=segment_counts)
    flat_sum = torch.zeros(num_voxels * vfe.num_bins, dtype=torch.float32, device=points.device)
    flat_sum[unique_segment] = segment_sums.float()
    flat_count = count_hist.view(-1)
    intensity_mean = torch.zeros((num_voxels, vfe.num_bins), dtype=torch.float32, device=points.device)
    occupied = flat_count > 0
    intensity_mean.view(-1)[occupied] = flat_sum[occupied] / flat_count[occupied].float()
    marks[3].record()

    centers = vfe._pillar_centers(voxel_coords).to(points.device)
    histogram_features = torch.cat((count_hist.float(), intensity_mean.float(), centers.float()), dim=1)
    marks[4].record()
    pillar_features = vfe.projection(histogram_features.float()).float()
    marks[5].record()
    marks[5].synchronize()

    names = ['histogram_key_lookup', 'count_reduction', 'intensity_sum_mean', 'concat_center', 'projection']
    timings = {name: float(marks[i].elapsed_time(marks[i + 1])) for i, name in enumerate(names)}
    return count_hist, intensity_mean, pillar_features, timings


def component_profile(vfe, batch, repeats=30):
    samples = {name: [] for name in (
        'histogram_key_lookup', 'count_reduction', 'intensity_sum_mean', 'concat_center', 'projection'
    )}
    reference = vfe(clone_batch(batch))['pillar_features']
    last = None
    for _ in range(repeats):
        _, _, last, timing = component_pass(vfe, batch)
        for name, value in timing.items():
            samples[name].append(value)
    torch.testing.assert_close(last, reference, rtol=0, atol=0)
    summary = {name: stats(values) for name, values in samples.items()}
    total = sum(value['p50_ms'] for value in summary.values())
    for value in summary.values():
        value['p50_fraction'] = value['p50_ms'] / total if total else 0.0
    return {'components': summary, 'raw_samples_ms': samples, 'exact_reference_match': True}


def build_synthetic(device, overflow=False):
    if overflow:
        count = 40000
        points = torch.zeros((count, 5), dtype=torch.float32, device=device)
        points[:, 1] = 0.01
        points[:, 2] = -39.67
        points[:, 3] = torch.linspace(-2.999, 0.999, count, device=device)
        points[:, 4] = torch.linspace(0.0, 1.0, count, device=device)
    else:
        points = torch.tensor([
            [0, 0.01, -39.67, -2.999, 0.1],
            [0, 0.02, -39.66, 0.999, 0.9],
        ], dtype=torch.float32, device=device)
    return {
        'points': points,
        'voxel_coords': torch.tensor([[0, 0, 0, 0]], dtype=torch.float32, device=device),
        'batch_size': 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('formal R6 profile requires CUDA')
    set_determinism(args.seed)

    repo_root = Path(__file__).resolve().parents[2]

    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu

    config = EasyDict()
    cfg_from_yaml_file('cfgs/kitti_models/pointpillar_pillarhist.yaml', config)

    def dataset(training):
        value, _, _ = build_dataloader(
            dataset_cfg=config.DATA_CONFIG, class_names=config.CLASS_NAMES,
            batch_size=2, dist=False, workers=0, logger=None,
            training=training, seed=args.seed,
        )
        return value

    validation_dataset = dataset(False)
    training_dataset = dataset(True)
    validation_batch = make_batch(validation_dataset, ['000001', '000002'])
    training_batch = make_batch(training_dataset, ['000000', '000003'])
    load_data_to_gpu(validation_batch)
    load_data_to_gpu(training_batch)

    model = build_network(config.MODEL, len(config.CLASS_NAMES), validation_dataset)
    checkpoint_path = repo_root / 'output/kitti_models/pointpillar/full_80ep_bs2_seed666_run001/ckpt/checkpoint_epoch_80.pth'
    backend_init = {'source': None, 'copied_tensors': 0}
    if checkpoint_path.exists():
        from pcdet.models.detectors.detector3d_template import _load_checkpoint
        checkpoint = _load_checkpoint(str(checkpoint_path), map_location='cpu')
        current = model.state_dict()
        compatible = {
            key: value for key, value in checkpoint['model_state'].items()
            if not key.startswith('vfe.') and key in current and current[key].shape == value.shape
        }
        current.update(compatible)
        model.load_state_dict(current, strict=True)
        backend_init = {
            'source': str(checkpoint_path),
            'sha256': sha256(checkpoint_path),
            'loader': 'pcdet.models.detectors.detector3d_template._load_checkpoint',
            'copied_tensors': len(compatible),
        }
    model.cuda().eval()
    vfe = model.vfe

    fixtures = {
        'synthetic_boundary': build_synthetic('cuda', False),
        'synthetic_overflow_40000': build_synthetic('cuda', True),
        'validation_000001_000002': validation_batch,
        'training_000000_000003': training_batch,
    }
    report = {
        'schema_version': 'pillarhist-r6-reference-profile-v1',
        'status': 'FORMAL',
        'started_utc': utc_now(),
        'seed': args.seed,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status_tracked_porcelain': subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], text=True
        ).splitlines(),
        'reduction_mode': vfe.reduction_mode,
        'determinism': {
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        },
        'device': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'backend_initialization': backend_init,
        'fixtures': {},
    }

    with torch.no_grad():
        for name, batch in fixtures.items():
            entry = {
                'points': int(batch['points'].shape[0]),
                'active_pillars': int(batch['voxel_coords'].shape[0]),
                'component_profile': component_profile(vfe, batch),
                'vfe': measure_cuda(lambda b=batch: vfe(clone_batch(b)), warmup=10, iterations=50),
            }
            if name.startswith(('validation_', 'training_')):
                entry['end_to_end_inference'] = measure_cuda(
                    lambda b=batch: model(clone_batch(b)), warmup=3, iterations=10
                )
            report['fixtures'][name] = entry

    report['finished_utc'] = utc_now()
    real_profiles = [
        report['fixtures']['validation_000001_000002']['component_profile']['components'],
        report['fixtures']['training_000000_000003']['component_profile']['components'],
    ]
    aggregate = {
        name: statistics.fmean(profile[name]['p50_fraction'] for profile in real_profiles)
        for name in real_profiles[0]
    }
    report['bottleneck_fraction_real_fixture_mean'] = aggregate
    report['primary_bottleneck'] = max(aggregate, key=aggregate.get)
    report['interpretation_rule'] = 'largest mean p50 component fraction across the two frozen real fixtures'

    args.output_root.mkdir(parents=True, exist_ok=False)
    output = args.output_root / 'reference_profile.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'output': str(output),
        'primary_bottleneck': report['primary_bottleneck'],
        'fractions': aggregate,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
