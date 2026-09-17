#!/usr/bin/env python3
"""Run one frozen PillarHist R7 short-training track with exact replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from prepare_pillarhist_r7 import (
    TRACK_SPECS,
    apply_canonical_initialization,
    batch_checksums,
    build_dataset,
    configure_track,
    load_config,
    materialize_replay_batch,
    set_seed,
    sha256_file,
    tensor_dict_checksum,
    to_plain,
    utc_now,
)


def json_dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain(value), indent=2), encoding='utf-8')


def db_sampler_state(dataset):
    states = {}
    for index, augmentor in enumerate(dataset.data_augmentor.data_augmentor_queue):
        if hasattr(augmentor, 'sample_groups'):
            states[str(index)] = copy.deepcopy(augmentor.sample_groups)
    return states


def restore_db_sampler_state(dataset, states):
    for index, augmentor in enumerate(dataset.data_augmentor.data_augmentor_queue):
        if str(index) in states:
            augmentor.sample_groups = copy.deepcopy(states[str(index)])


def rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])


def cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def save_checkpoint(path, model, optimizer, scaler, dataset, track, completed_steps,
                    replay_sha256, canonical_sha256, total_steps):
    state = {
        'schema_version': 'pillarhist-r7-checkpoint-v1',
        'track': track,
        'completed_optimizer_steps': completed_steps,
        'next_step_zero_based': completed_steps,
        'total_steps': total_steps,
        'model_state': cpu_state_dict(model),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': {
            'type': 'OpenPCDet OneCycle',
            'next_step_zero_based': completed_steps,
            'total_steps': total_steps,
        },
        'grad_scaler_state': scaler.state_dict(),
        'amp_enabled': False,
        'rng_state': rng_state(),
        'db_sampler_state': db_sampler_state(dataset),
        'sampler_cursor_samples': completed_steps * 2,
        'accumulation_state': {'enabled': False, 'micro_batches_pending': 0},
        'training_replay_manifest_sha256': replay_sha256,
        'canonical_initialization_sha256': canonical_sha256,
        'saved_utc': utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return state


def load_checkpoint(path, model, optimizer, scaler, dataset, replay_sha256,
                    canonical_sha256):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['training_replay_manifest_sha256'] != replay_sha256:
        raise AssertionError('resume replay manifest hash mismatch')
    if checkpoint['canonical_initialization_sha256'] != canonical_sha256:
        raise AssertionError('resume canonical initialization hash mismatch')
    model.load_state_dict(checkpoint['model_state'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scaler.load_state_dict(checkpoint['grad_scaler_state'])
    restore_rng_state(checkpoint['rng_state'])
    restore_db_sampler_state(dataset, checkpoint['db_sampler_state'])
    return checkpoint


def verify_replay_entry(dataset, entry):
    batch, matrices, frames = materialize_replay_batch(dataset, entry)
    checksums = batch_checksums(batch)
    failures = {}
    if frames != entry['actual_frame_ids']:
        failures['frame_ids'] = {'actual': frames, 'expected': entry['actual_frame_ids']}
    if matrices != entry['augmentation_matrices']:
        failures['augmentation_matrices'] = True
    for key, expected in entry['checksums'].items():
        if checksums[key] != expected:
            failures[key] = {'actual': checksums[key], 'expected': expected}
    if failures:
        raise AssertionError({'replay_step': entry['step_zero_based'], 'failures': failures})
    return batch, checksums


def finite_number(value):
    return math.isfinite(float(value))


def representative_gradients(model):
    backbone_parameter = next(
        parameter for parameter in model.backbone_2d.blocks[0].parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    )
    tensors = {
        'backbone': backbone_parameter.grad,
        'head': model.dense_head.conv_cls.weight.grad,
    }
    if hasattr(model.vfe, 'projection'):
        projection = model.vfe.projection
        tensors['projection'] = (
            projection.weight.grad if isinstance(projection, torch.nn.Linear)
            else projection[0].weight.grad
        )
    result = {}
    for name, gradient in tensors.items():
        result[name] = {
            'present': gradient is not None,
            'finite': bool(gradient is not None and torch.isfinite(gradient).all()),
            'nonzero': bool(gradient is not None and torch.count_nonzero(gradient) > 0),
            'norm': float(gradient.norm().detach().cpu()) if gradient is not None else None,
        }
    result['pass'] = all(
        item['present'] and item['finite'] and item['nonzero']
        for name, item in result.items() if name != 'pass'
    )
    return result


def make_optimizer_and_scheduler(model, optim_cfg, total_steps):
    from train_utils.optimization import build_optimizer, build_scheduler
    optimizer = build_optimizer(model, optim_cfg)
    scheduler, warmup = build_scheduler(
        optimizer=optimizer,
        total_iters_each_epoch=total_steps,
        total_epochs=1,
        last_epoch=-1,
        optim_cfg=optim_cfg,
    )
    if warmup is not None:
        raise AssertionError('R7 protocol forbids additional LR warmup')
    return optimizer, scheduler


def loss_only(model, batch):
    from pcdet.models import load_data_to_gpu
    batch = copy.deepcopy(batch)
    load_data_to_gpu(batch)
    model.train()
    with torch.no_grad():
        result, _, _ = model(batch)
    return float(result['loss'].detach().cpu())


def resume_continuity_check(checkpoint_path, model, optimizer, scaler, dataset,
                            replay_entry, replay_sha256, canonical_sha256):
    checkpoint = load_checkpoint(
        checkpoint_path, model, optimizer, scaler, dataset,
        replay_sha256, canonical_sha256,
    )
    expected_batch, expected_checksums = verify_replay_entry(dataset, replay_entry)
    expected_loss = loss_only(model, expected_batch)

    load_checkpoint(
        checkpoint_path, model, optimizer, scaler, dataset,
        replay_sha256, canonical_sha256,
    )
    actual_batch, actual_checksums = verify_replay_entry(dataset, replay_entry)
    actual_loss = loss_only(model, actual_batch)
    passed = (
        expected_checksums == actual_checksums
        and math.isclose(actual_loss, expected_loss, rel_tol=1e-5, abs_tol=1e-6)
    )
    result = {
        'checkpoint': str(checkpoint_path),
        'completed_optimizer_steps': checkpoint['completed_optimizer_steps'],
        'next_step_zero_based': replay_entry['step_zero_based'],
        'expected_input_checksum': expected_checksums['input'],
        'actual_input_checksum': actual_checksums['input'],
        'expected_loss': expected_loss,
        'actual_loss': actual_loss,
        'rtol': 1e-5,
        'atol': 1e-6,
        'pass': passed,
    }
    load_checkpoint(
        checkpoint_path, model, optimizer, scaler, dataset,
        replay_sha256, canonical_sha256,
    )
    if not passed:
        raise AssertionError(result)
    return result


@torch.no_grad()
def evaluate(model, dataset, indices, config, output_dir, label):
    from pcdet.models import load_data_to_gpu
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    class_names = list(config.CLASS_NAMES)
    det_annos = []
    prediction_count = 0
    started = time.time()
    for offset in range(0, len(indices), 2):
        chunk = indices[offset:offset + 2]
        batch = dataset.collate_batch([dataset[index] for index in chunk])
        load_data_to_gpu(batch)
        predictions, _ = model(batch)
        prediction_count += sum(int(item['pred_boxes'].shape[0]) for item in predictions)
        det_annos.extend(dataset.generate_prediction_dicts(batch, predictions, class_names))
        if offset % 100 == 0:
            print(json.dumps({
                'phase': 'validation', 'label': label,
                'frames_done': min(offset + len(chunk), len(indices)),
                'frames_total': len(indices), 'utc': utc_now(),
            }), flush=True)

    original_infos = dataset.kitti_infos
    try:
        dataset.kitti_infos = [original_infos[index] for index in indices]
        result_text, metrics = dataset.evaluation(
            det_annos, class_names,
            eval_metric=config.MODEL.POST_PROCESSING.EVAL_METRIC,
        )
    finally:
        dataset.kitti_infos = original_infos
    metrics = {key: float(value) for key, value in metrics.items()}
    moderate = {
        name: metrics[f'{name}_3d/moderate_R40'] for name in class_names
    }
    summary = {
        'label': label,
        'frames': len(indices),
        'prediction_count': prediction_count,
        'three_class_3d_ap_r40_moderate': moderate,
        'macro_3d_ap_r40_moderate': statistics.fmean(moderate.values()),
        'elapsed_seconds': time.time() - started,
        'metrics': metrics,
    }
    json_dump(output_dir / 'metrics.json', summary)
    (output_dir / 'result.txt').write_text(result_text, encoding='utf-8')
    with (output_dir / 'result.pkl').open('wb') as stream:
        pickle.dump(det_annos, stream)
    model.train()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation-root', type=Path, required=True)
    parser.add_argument('--track', choices=TRACK_SPECS, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--validation-interval', type=int, default=1000)
    parser.add_argument('--resume-check-step', type=int, default=1000)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--skip-evaluation', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('R7 training requires CUDA')
    args.output_root.mkdir(parents=True, exist_ok=True)

    replay_path = args.preparation_root / 'training_replay_manifest.json'
    canonical_path = args.preparation_root / 'canonical_initialization.pth'
    replay_sha256 = sha256_file(replay_path)
    canonical_sha256 = sha256_file(canonical_path)
    replay = json.loads(replay_path.read_text(encoding='utf-8'))
    if args.steps > replay['training_steps']:
        raise ValueError('requested steps exceed frozen replay plan')
    package = torch.load(canonical_path, map_location='cpu', weights_only=False)

    repo = Path(__file__).resolve().parents[2]
    ph_config = load_config(repo / 'tools/cfgs/kitti_models/pointpillar_pillarhist.yaml')
    pp_config = load_config(repo / 'tools/cfgs/kitti_models/pointpillar.yaml')
    config = configure_track(ph_config, pp_config, args.track)
    seed = int(replay['seed'])
    train_dataset = build_dataset(ph_config, training=True, seed=seed)
    validation_dataset = build_dataset(ph_config, training=False, seed=seed)

    from pcdet.models import build_network, load_data_to_gpu
    set_seed(seed, cuda=True)
    model = build_network(config.MODEL, len(config.CLASS_NAMES), train_dataset)
    initialization = apply_canonical_initialization(model, package, args.track)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, config.OPTIMIZATION, args.steps
    )
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    model.cuda().train()

    start_step = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume, model, optimizer, scaler, train_dataset,
            replay_sha256, canonical_sha256,
        )
        start_step = int(checkpoint['completed_optimizer_steps'])
    elif any(args.output_root.iterdir()):
        raise FileExistsError(f'non-empty output root without --resume: {args.output_root}')

    resolved = to_plain(config)
    (args.output_root / 'config_resolved.yaml').write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding='utf-8'
    )
    run_manifest = {
        'schema_version': 'pillarhist-r7-track-run-v1',
        'track': args.track,
        'started_utc': utc_now(),
        'seed': seed,
        'steps': args.steps,
        'batch_size': 2,
        'workers': 0,
        'amp': False,
        'gradient_accumulation': False,
        'reduction_path': (
            model.vfe.reduction_mode if hasattr(model.vfe, 'reduction_mode') else 'PillarVFE'
        ),
        'optimizer_created_after_canonical_copy': True,
        'initialization': initialization,
        'replay_manifest_sha256': replay_sha256,
        'canonical_initialization_sha256': canonical_sha256,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'runner_sha256': sha256_file(__file__),
        'resume_checkpoint': str(args.resume) if args.resume else None,
    }
    json_dump(args.output_root / 'manifest.json', run_manifest)

    subset_spec = json.loads((repo / 'docs/pillarhist/PillarHist_R7_validation_subset_v1.3.json').read_text(encoding='utf-8'))
    subset_indices = subset_spec['selection']['source_indices_zero_based']
    expected_subset_frames = subset_spec['selection']['frame_ids']
    actual_subset_frames = [
        str(validation_dataset.kitti_infos[index]['point_cloud']['lidar_idx'])
        for index in subset_indices
    ]
    if actual_subset_frames != expected_subset_frames:
        raise AssertionError('validation subset frame IDs do not match frozen manifest')

    losses = []
    gradient_checks = []
    validations = []
    resume_result = None
    peak_secondary = None
    log_path = args.output_root / 'training_log.jsonl'
    mode = 'a' if start_step else 'w'
    start_time = time.time()
    with log_path.open(mode, encoding='utf-8') as log_stream:
        for step in range(start_step, args.steps):
            batch, checksums = verify_replay_entry(train_dataset, replay['entries'][step])
            load_data_to_gpu(batch)
            scheduler.step(step, 0)
            current_lr = float(optimizer.lr)
            optimizer.zero_grad()
            model.train()
            result, tb, _ = model(batch)
            loss = result['loss']
            positive_anchors = int((model.dense_head.forward_ret_dict['box_cls_labels'] > 0).sum())
            components = {key: float(value) for key, value in tb.items()}
            if not torch.isfinite(loss) or not all(finite_number(value) for value in components.values()):
                raise FloatingPointError({'step': step, 'loss': float(loss.detach().cpu()), 'components': components})
            if positive_anchors <= 0:
                raise AssertionError({'step': step, 'positive_anchors': positive_anchors})
            loss.backward()
            gradient_check = None
            if (step + 1) % 100 == 0:
                gradient_check = representative_gradients(model)
                if not gradient_check['pass']:
                    raise AssertionError({'step': step + 1, 'gradient_check': gradient_check})
                gradient_checks.append({'step': step + 1, **gradient_check})
            grad_norm = float(clip_grad_norm_(model.parameters(), config.OPTIMIZATION.GRAD_NORM_CLIP).detach().cpu())
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            record = {
                'step': step + 1,
                'loss': loss_value,
                'loss_components': components,
                'lr': current_lr,
                'gradient_norm_before_clip': grad_norm,
                'positive_anchors': positive_anchors,
                'input_checksum': checksums['input'],
                'gradient_check': gradient_check,
                'elapsed_seconds': time.time() - start_time,
            }
            log_stream.write(json.dumps(record) + '\n')
            log_stream.flush()

            checkpoint_due = (step + 1) % args.validation_interval == 0 or step + 1 == args.steps
            if checkpoint_due:
                checkpoint_path = args.output_root / 'checkpoints' / f'step_{step + 1:05d}.pth'
                save_checkpoint(
                    checkpoint_path, model, optimizer, scaler, train_dataset,
                    args.track, step + 1, replay_sha256, canonical_sha256, args.steps,
                )
                if step + 1 == args.resume_check_step and step + 1 < len(replay['entries']):
                    resume_result = resume_continuity_check(
                        checkpoint_path, model, optimizer, scaler, train_dataset,
                        replay['entries'][step + 1], replay_sha256, canonical_sha256,
                    )
                    json_dump(args.output_root / 'diagnostics/resume_continuity.json', resume_result)
                if not args.skip_evaluation:
                    subset_result = evaluate(
                        model, validation_dataset, subset_indices, config,
                        args.output_root / 'validation' / f'step_{step + 1:05d}_subset',
                        f'step_{step + 1:05d}_subset',
                    )
                    validations.append({'step': step + 1, **subset_result})
                    if peak_secondary is None or subset_result['macro_3d_ap_r40_moderate'] > peak_secondary['macro_3d_ap_r40_moderate']:
                        peak_secondary = {'step': step + 1, **subset_result}
            if (step + 1) % 100 == 0 or step == start_step:
                print(json.dumps({
                    'phase': 'train', 'track': args.track, 'step': step + 1,
                    'loss': loss_value, 'lr': current_lr,
                    'elapsed_seconds': time.time() - start_time, 'utc': utc_now(),
                }), flush=True)

    full_validation = None
    if not args.skip_evaluation:
        full_validation = evaluate(
            model, validation_dataset, list(range(len(validation_dataset))), config,
            args.output_root / 'validation' / f'step_{args.steps:05d}_full',
            f'step_{args.steps:05d}_full',
        )
    final_checkpoint = args.output_root / 'checkpoints' / f'step_{args.steps:05d}.pth'
    shutil.copy2(final_checkpoint, args.output_root / 'checkpoints' / 'last.pth')
    if peak_secondary is not None:
        peak_checkpoint = args.output_root / 'checkpoints' / f"step_{peak_secondary['step']:05d}.pth"
        shutil.copy2(peak_checkpoint, args.output_root / 'checkpoints' / 'peak_secondary.pth')
    first_window = losses[:min(200, len(losses))]
    last_window = losses[-min(200, len(losses)):]
    summary = {
        'track': args.track,
        'status': 'COMPLETE',
        'completed_optimizer_steps': args.steps,
        'finished_utc': utc_now(),
        'loss': {
            'first_200_median': statistics.median(first_window),
            'last_200_median': statistics.median(last_window),
            'last_below_first': statistics.median(last_window) < statistics.median(first_window),
            'minimum': min(losses),
            'maximum': max(losses),
        },
        'gradient_checks': gradient_checks,
        'all_gradient_checks_pass': all(item['pass'] for item in gradient_checks),
        'resume_continuity': resume_result,
        'subset_validations': validations,
        'peak_secondary': peak_secondary,
        'full_validation_primary': full_validation,
        'canonical_initial_model_checksum': initialization['complete_model_checksum'],
        'final_model_checksum': tensor_dict_checksum(model.state_dict()),
    }
    if args.track == 'PP_SHORT_CONTROL' and not args.skip_evaluation:
        full = full_validation
        summary['pp_short_control_sanity'] = {
            'loss_finite_and_positive_anchors': True,
            'representative_gradients': summary['all_gradient_checks_pass'],
            'last_200_median_below_first_200': summary['loss']['last_below_first'],
            'full_validation_nonempty_predictions': full['prediction_count'] > 0,
            'car_ap_above_zero': full['three_class_3d_ap_r40_moderate']['Car'] > 0,
            'macro_ap_above_zero': full['macro_3d_ap_r40_moderate'] > 0,
            'checkpoint_resume': bool(resume_result and resume_result['pass']),
        }
        summary['pp_short_control_sanity']['pass'] = all(
            summary['pp_short_control_sanity'].values()
        )
    json_dump(args.output_root / 'track_summary.json', summary)
    print(json.dumps({
        'track': args.track,
        'status': summary['status'],
        'loss': summary['loss'],
        'full_macro': full_validation['macro_3d_ap_r40_moderate'] if full_validation else None,
    }, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
