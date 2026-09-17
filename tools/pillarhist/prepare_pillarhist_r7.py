#!/usr/bin/env python3
"""Prepare canonical initialization and a fully checksummed R7 replay plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict


TRACK_SPECS = {
    'PH_RAW_LINEAR': {'family': 'pillarhist', 'coord': 'raw_meter_xy', 'projection': 'linear'},
    'PH_NORM_LINEAR': {'family': 'pillarhist', 'coord': 'normalized_xy', 'projection': 'linear'},
    'PH_RAW_BNRELU': {'family': 'pillarhist', 'coord': 'raw_meter_xy', 'projection': 'linear_bn_relu'},
    'PH_NORM_BNRELU': {'family': 'pillarhist', 'coord': 'normalized_xy', 'projection': 'linear_bn_relu'},
    'PP_SHORT_CONTROL': {'family': 'pointpillar'},
}


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def set_seed(seed, cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def update_hash(digest, value, name):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(name.encode('utf-8'))
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(json.dumps(list(array.shape)).encode('ascii'))
        digest.update(array.tobytes())
    elif isinstance(value, (list, tuple)) and all(isinstance(item, (str, np.str_)) for item in value):
        digest.update(name.encode('utf-8'))
        digest.update(json.dumps([str(item) for item in value], ensure_ascii=False).encode('utf-8'))
    elif isinstance(value, (str, int, float, bool)):
        digest.update(name.encode('utf-8'))
        digest.update(repr(value).encode('utf-8'))


def values_checksum(named_values):
    digest = hashlib.sha256()
    for name, value in sorted(named_values.items()):
        update_hash(digest, value, name)
    return digest.hexdigest()


def batch_checksums(batch):
    numeric = {
        key: value for key, value in batch.items()
        if torch.is_tensor(value) or isinstance(value, np.ndarray)
    }
    frame_ids = [str(value) for value in np.asarray(batch['frame_id']).reshape(-1)]
    complete = dict(numeric)
    complete['frame_id'] = frame_ids
    return {
        'input': values_checksum(complete),
        'points': values_checksum({'points': batch['points']}),
        'gt_database_effect': values_checksum({
            'gt_boxes': batch['gt_boxes'],
            'points': batch['points'],
        }),
        'voxelization': values_checksum({
            'voxels': batch['voxels'],
            'voxel_coords': batch['voxel_coords'],
            'voxel_num_points': batch['voxel_num_points'],
        }),
        'augmentation_parameters': values_checksum({
            'lidar_aug_matrix': batch['lidar_aug_matrix'],
        }),
    }


def load_config(path):
    from pcdet.config import cfg_from_yaml_file
    config = EasyDict()
    cfg_from_yaml_file(str(path), config)
    return config


def build_dataset(config, training, seed):
    from pcdet.datasets import build_dataloader
    set_seed(seed)
    dataset, _, _ = build_dataloader(
        dataset_cfg=config.DATA_CONFIG,
        class_names=config.CLASS_NAMES,
        batch_size=2,
        dist=False,
        workers=0,
        logger=None,
        training=training,
        seed=seed,
    )
    return dataset


def replay_schedule(dataset_size, batches, seed):
    order_rng = np.random.default_rng(seed + 17001)
    seed_rng = np.random.default_rng(seed + 29003)
    indices = []
    while len(indices) < batches * 2:
        indices.extend(order_rng.permutation(dataset_size).tolist())
    indices = indices[:batches * 2]
    seeds = seed_rng.integers(1, 2**31 - 1, size=batches * 2, dtype=np.int64).tolist()
    return [
        {
            'step_zero_based': step,
            'optimizer_step_one_based': step + 1 if step < batches - 1 else None,
            'role': 'training' if step < batches - 1 else 'resume_probe_after_step_5000',
            'dataset_indices': indices[2 * step:2 * step + 2],
            'sample_seeds': seeds[2 * step:2 * step + 2],
        }
        for step in range(batches)
    ]


def materialize_replay_batch(dataset, entry):
    samples = []
    augmentation_matrices = []
    actual_frames = []
    for index, sample_seed in zip(entry['dataset_indices'], entry['sample_seeds']):
        set_seed(int(sample_seed))
        sample = dataset[int(index)]
        samples.append(sample)
        augmentation_matrices.append(np.asarray(sample['lidar_aug_matrix']).tolist())
        actual_frames.append(str(sample['frame_id']))
    batch = dataset.collate_batch(samples)
    return batch, augmentation_matrices, actual_frames


def configure_track(base_config, pp_config, track):
    spec = TRACK_SPECS[track]
    config = copy.deepcopy(pp_config if spec['family'] == 'pointpillar' else base_config)
    config.OPTIMIZATION.BATCH_SIZE_PER_GPU = 2
    config.OPTIMIZATION.NUM_EPOCHS = 1
    if spec['family'] == 'pillarhist':
        config.MODEL.VFE.COORD_MODE = spec['coord']
        config.MODEL.VFE.REDUCTION_MODE = 'deterministic_segment'
        config.MODEL.VFE.PROJECTION.TYPE = spec['projection']
        config.MODEL.VFE.PROJECTION.BIAS = spec['projection'] == 'linear'
        config.MODEL.VFE.PROJECTION.BN_EPS = 0.001
        config.MODEL.VFE.PROJECTION.BN_MOMENTUM = 0.01
    return config


def tensor_dict_checksum(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        update_hash(digest, value, key)
    return digest.hexdigest()


def apply_canonical_initialization(model, package, track):
    spec = TRACK_SPECS[track]
    current = model.state_dict()
    copied_backend = []
    for key, value in package['backend_state'].items():
        if key not in current or current[key].shape != value.shape:
            raise AssertionError(f'canonical backend mismatch for {track}: {key}')
        current[key] = value.clone()
        copied_backend.append(key)
    model.load_state_dict(current, strict=True)

    if spec['family'] == 'pillarhist':
        with torch.no_grad():
            if spec['projection'] == 'linear':
                model.vfe.projection.weight.copy_(package['projection_weight'])
                model.vfe.projection.bias.copy_(package['projection_bias'])
            else:
                linear = model.vfe.projection[0]
                bn = model.vfe.projection[1]
                linear.weight.copy_(package['projection_weight'])
                bn.weight.fill_(1.0)
                bn.bias.zero_()
                bn.running_mean.zero_()
                bn.running_var.fill_(1.0)
                bn.num_batches_tracked.zero_()
                if linear.bias is not None:
                    raise AssertionError('BN-ReLU canonical Linear must use bias=False')
                if bn.eps != 0.001 or bn.momentum != 0.01:
                    raise AssertionError('BN canonical hyperparameters changed')
    backend_checksum = tensor_dict_checksum({
        key: model.state_dict()[key] for key in copied_backend
    })
    if backend_checksum != package['backend_checksum']:
        raise AssertionError(f'backend checksum mismatch after copy for {track}')
    return {
        'copied_backend_tensor_count': len(copied_backend),
        'backend_checksum': backend_checksum,
        'complete_model_checksum': tensor_dict_checksum(model.state_dict()),
        'projection_weight_checksum': (
            values_checksum({'weight': package['projection_weight']})
            if spec['family'] == 'pillarhist' else None
        ),
    }


def to_plain(value):
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--train-steps', type=int, default=5000)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)

    repo = Path(__file__).resolve().parents[2]
    ph_cfg_path = repo / 'tools/cfgs/kitti_models/pointpillar_pillarhist.yaml'
    pp_cfg_path = repo / 'tools/cfgs/kitti_models/pointpillar.yaml'
    ph_config = load_config(ph_cfg_path)
    pp_config = load_config(pp_cfg_path)

    train_dataset = build_dataset(ph_config, training=True, seed=args.seed)
    schedule = replay_schedule(len(train_dataset), args.train_steps + 1, args.seed)
    for entry in schedule:
        batch, matrices, frames = materialize_replay_batch(train_dataset, entry)
        entry['actual_frame_ids'] = frames
        entry['augmentation_matrices'] = matrices
        entry['checksums'] = batch_checksums(batch)
        if entry['step_zero_based'] % 250 == 0:
            print(json.dumps({'phase': 'replay', 'step': entry['step_zero_based'], 'utc': utc_now()}), flush=True)

    source_paths = {
        'train_infos': repo / 'data/kitti/kitti_infos_train.pkl',
        'gt_database_infos': repo / 'data/kitti/kitti_dbinfos_train.pkl',
        'train_split': repo / 'data/kitti/ImageSets/train.txt',
    }
    audit_rng = np.random.default_rng(args.seed)
    interior_population = np.arange(1, max(1, args.train_steps - 1))
    interior_count = min(20, len(interior_population))
    interior = sorted(
        audit_rng.choice(interior_population, size=interior_count, replace=False).tolist()
    ) if interior_count else []
    replay_manifest = {
        'schema_version': 'pillarhist-r7-training-replay-v1',
        'created_utc': utc_now(),
        'seed': args.seed,
        'batch_size': 2,
        'workers': 0,
        'training_steps': args.train_steps,
        'resume_probe_steps': 1,
        'dataset_size': len(train_dataset),
        'sources': {
            name: {'path': str(path), 'sha256': sha256_file(path)}
            for name, path in source_paths.items()
        },
        'audit_step_indices_zero_based': [0, *interior, args.train_steps - 1],
        'entries': schedule,
    }
    replay_path = args.output_root / 'training_replay_manifest.json'
    replay_path.write_text(json.dumps(replay_manifest, indent=2), encoding='utf-8')

    from pcdet.models import build_network
    set_seed(args.seed, cuda=False)
    canonical_model = build_network(
        ph_config.MODEL, len(ph_config.CLASS_NAMES), train_dataset
    )
    canonical_state = canonical_model.state_dict()
    backend_state = {
        key: value.detach().cpu().clone()
        for key, value in canonical_state.items() if not key.startswith('vfe.')
    }
    package = {
        'schema_version': 'pillarhist-r7-canonical-init-v1',
        'seed': args.seed,
        'backend_state': backend_state,
        'backend_checksum': tensor_dict_checksum(backend_state),
        'projection_weight': canonical_model.vfe.projection.weight.detach().cpu().clone(),
        'projection_bias': canonical_model.vfe.projection.bias.detach().cpu().clone(),
    }
    canonical_path = args.output_root / 'canonical_initialization.pth'
    torch.save(package, canonical_path)

    initialization = {}
    for track in TRACK_SPECS:
        set_seed(args.seed)
        config = configure_track(ph_config, pp_config, track)
        model = build_network(config.MODEL, len(config.CLASS_NAMES), train_dataset)
        initialization[track] = apply_canonical_initialization(model, package, track)
        del model
    init_manifest = {
        'schema_version': package['schema_version'],
        'created_utc': utc_now(),
        'seed': args.seed,
        'optimizer_created_after_copy': True,
        'backend_checksum': package['backend_checksum'],
        'canonical_file': str(canonical_path),
        'canonical_file_sha256': sha256_file(canonical_path),
        'tracks': initialization,
    }
    init_path = args.output_root / 'canonical_initialization.json'
    init_path.write_text(json.dumps(init_manifest, indent=2), encoding='utf-8')

    manifest = {
        'schema_version': 'pillarhist-r7-preparation-v1',
        'status': 'PREPARED',
        'created_utc': utc_now(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest(),
        'source_sha256': {
            'prepare_pillarhist_r7.py': sha256_file(__file__),
            'run_pillarhist_r7_track.py': sha256_file(Path(__file__).with_name('run_pillarhist_r7_track.py')),
        },
        'seed': args.seed,
        'train_steps': args.train_steps,
        'tracks': list(TRACK_SPECS),
        'reduction_path': 'deterministic_segment',
        'replay_manifest': {'path': str(replay_path), 'sha256': sha256_file(replay_path)},
        'canonical_initialization': {'path': str(init_path), 'sha256': sha256_file(init_path)},
        'canonical_state': {'path': str(canonical_path), 'sha256': sha256_file(canonical_path)},
        'optimizer_policy': 'optimizer/scheduler creation is forbidden until apply_canonical_initialization returns',
    }
    (args.output_root / 'preparation_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    (args.output_root / 'resolved_track_configs.yaml').write_text(
        yaml.safe_dump({
            track: to_plain(configure_track(ph_config, pp_config, track))
            for track in TRACK_SPECS
        }, sort_keys=False),
        encoding='utf-8',
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
