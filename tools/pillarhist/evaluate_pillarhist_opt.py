#!/usr/bin/env python3
"""Formal TST-018 audit and pre-registered R6 REF/OPT benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict


HIST_RTOL = 1e-5
HIST_ATOL = 1e-6
GRAD_RTOL = 1e-4
GRAD_ATOL = 1e-6
GRAD_COSINE_MIN = 0.9999
REPEATS = 20


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def set_mode(seed, deterministic_algorithms=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic_algorithms)


def clone_batch(batch):
    return {
        key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in batch.items()
    }


def make_batch(dataset, frame_ids):
    by_id = {
        str(info['point_cloud']['lidar_idx']): index
        for index, info in enumerate(dataset.kitti_infos)
    }
    batch = dataset.collate_batch([dataset[by_id[frame_id]] for frame_id in frame_ids])
    actual = [str(value) for value in np.asarray(batch['frame_id']).reshape(-1)]
    if actual != frame_ids:
        raise AssertionError(f'fixture mismatch: expected {frame_ids}, got {actual}')
    return batch


def tensor_error(actual, expected, rtol, atol, exact=False):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            'pass': False,
            'shape_actual': list(actual.shape),
            'shape_expected': list(expected.shape),
            'dtype_actual': str(actual.dtype),
            'dtype_expected': str(expected.dtype),
        }
    if actual.numel() == 0:
        return {'pass': True, 'max_abs': 0.0, 'max_rel': 0.0, 'exact': True}
    a = actual.detach().float()
    e = expected.detach().float()
    delta = (a - e).abs()
    denominator = e.abs().clamp_min(max(atol, torch.finfo(torch.float32).eps))
    return {
        'pass': bool(torch.equal(actual, expected) if exact else torch.allclose(actual, expected, rtol=rtol, atol=atol)),
        'max_abs': float(delta.max().cpu()),
        'max_rel': float((delta / denominator).max().cpu()),
        'exact': bool(torch.equal(actual, expected)),
    }


def cosine(actual, expected):
    actual = actual.detach().float().flatten()
    expected = expected.detach().float().flatten()
    actual_norm = float(actual.norm().cpu())
    expected_norm = float(expected.norm().cpu())
    value = None
    if actual_norm > 1e-8 and expected_norm > 1e-8:
        value = float(torch.nn.functional.cosine_similarity(actual, expected, dim=0).cpu())
    return {'actual_norm': actual_norm, 'expected_norm': expected_norm, 'cosine': value}


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def sample_stats(values):
    return {
        'n': len(values),
        'min': min(values),
        'p50': percentile(values, 0.5),
        'p95': percentile(values, 0.95),
        'max': max(values),
        'mean': statistics.fmean(values),
    }


def build_context(seed):
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.models.detectors.detector3d_template import _load_checkpoint

    config = EasyDict()
    cfg_from_yaml_file('cfgs/kitti_models/pointpillar_pillarhist.yaml', config)

    def build_dataset(training):
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

    val_dataset = build_dataset(False)
    train_dataset = build_dataset(True)
    val_batch = make_batch(val_dataset, ['000001', '000002'])
    train_batch = make_batch(train_dataset, ['000000', '000003'])
    load_data_to_gpu(val_batch)
    load_data_to_gpu(train_batch)

    reference = build_network(config.MODEL, len(config.CLASS_NAMES), val_dataset)
    checkpoint_path = Path(__file__).resolve().parents[2] / (
        'output/kitti_models/pointpillar/full_80ep_bs2_seed666_run001/'
        'ckpt/checkpoint_epoch_80.pth'
    )
    checkpoint = _load_checkpoint(str(checkpoint_path), map_location='cpu')
    state = reference.state_dict()
    compatible = {
        key: value for key, value in checkpoint['model_state'].items()
        if not key.startswith('vfe.') and key in state and state[key].shape == value.shape
    }
    state.update(compatible)
    reference.load_state_dict(state, strict=True)

    opt_cfg = copy.deepcopy(config.MODEL)
    opt_cfg.VFE.REDUCTION_MODE = 'compact_lookup_segment'
    optimized = build_network(opt_cfg, len(config.CLASS_NAMES), val_dataset)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    reference.cuda()
    optimized.cuda()
    canonical_state = {
        key: value.detach().cpu().clone()
        for key, value in reference.state_dict().items()
    }
    return {
        'config': config,
        'reference': reference,
        'optimized': optimized,
        'canonical_state': canonical_state,
        'validation_batch': val_batch,
        'training_batch': train_batch,
        'checkpoint': {
            'path': str(checkpoint_path),
            'sha256': sha256(checkpoint_path),
            'compatible_backend_tensors': len(compatible),
        },
    }


def restore_models(context, train):
    for model in (context['reference'], context['optimized']):
        model.load_state_dict(context['canonical_state'], strict=True)
        model.zero_grad(set_to_none=True)
        model.train(train)


@torch.no_grad()
def compare_eval(context, name, batch):
    restore_models(context, train=False)
    ref = context['reference']
    opt = context['optimized']
    ref_count, ref_intensity = ref.vfe.build_histograms(
        batch['points'], batch['voxel_coords'], batch['batch_size']
    )
    opt_count, opt_intensity = opt.vfe.build_histograms(
        batch['points'], batch['voxel_coords'], batch['batch_size']
    )
    ref_features = ref.vfe(clone_batch(batch))['pillar_features']
    opt_features = opt.vfe(clone_batch(batch))['pillar_features']
    ref_pred, _ = ref(clone_batch(batch))
    ref_head = {
        key: value.detach().clone()
        for key, value in ref.dense_head.forward_ret_dict.items()
        if key in ('cls_preds', 'box_preds', 'dir_cls_preds')
    }
    opt_pred, _ = opt(clone_batch(batch))
    opt_head = {
        key: value.detach().clone()
        for key, value in opt.dense_head.forward_ret_dict.items()
        if key in ('cls_preds', 'box_preds', 'dir_cls_preds')
    }
    checks = {
        'Hp_count': tensor_error(opt_count, ref_count, 0, 0, exact=True),
        'Hi_intensity_mean': tensor_error(opt_intensity, ref_intensity, HIST_RTOL, HIST_ATOL),
        'pillar_features': tensor_error(opt_features, ref_features, HIST_RTOL, HIST_ATOL),
    }
    for key in ref_head:
        checks[f'head_{key}'] = tensor_error(opt_head[key], ref_head[key], HIST_RTOL, HIST_ATOL)
    post = []
    for index, (actual, expected) in enumerate(zip(opt_pred, ref_pred)):
        item = {
            'boxes': tensor_error(actual['pred_boxes'], expected['pred_boxes'], HIST_RTOL, HIST_ATOL),
            'scores': tensor_error(actual['pred_scores'], expected['pred_scores'], HIST_RTOL, HIST_ATOL),
            'labels': tensor_error(actual['pred_labels'], expected['pred_labels'], 0, 0, exact=True),
        }
        checks[f'post_nms_{index}'] = {'pass': all(value['pass'] for value in item.values()), **item}
        post.append(item)
    return {
        'fixture': name,
        'frames': [str(value) for value in np.asarray(batch['frame_id']).reshape(-1)],
        'points': int(batch['points'].shape[0]),
        'active_pillars': int(batch['voxel_coords'].shape[0]),
        'checks': checks,
        'pass': all(value['pass'] for value in checks.values()),
    }


def train_backward(model, batch):
    model.zero_grad(set_to_none=True)
    result, tb, _ = model(clone_batch(batch))
    loss = result['loss']
    loss.backward()
    return {
        'loss': loss.detach().clone(),
        'loss_components': {key: float(value) for key, value in tb.items()},
        'projection_grad': model.vfe.projection.weight.grad.detach().clone(),
        'backend_grad': model.dense_head.conv_cls.weight.grad.detach().clone(),
        'positive_anchors': int((model.dense_head.forward_ret_dict['box_cls_labels'] > 0).sum()),
    }


def repeatability_audit(context):
    batch = context['training_batch']
    restore_models(context, train=True)
    expected = train_backward(context['reference'], batch)
    records = []
    passed = True
    for repeat in range(REPEATS):
        context['optimized'].load_state_dict(context['canonical_state'], strict=True)
        context['optimized'].train()
        actual = train_backward(context['optimized'], batch)
        loss_error = tensor_error(actual['loss'], expected['loss'], HIST_RTOL, HIST_ATOL)
        projection_error = tensor_error(
            actual['projection_grad'], expected['projection_grad'], GRAD_RTOL, GRAD_ATOL
        )
        backend_error = tensor_error(
            actual['backend_grad'], expected['backend_grad'], GRAD_RTOL, GRAD_ATOL
        )
        projection_cosine = cosine(actual['projection_grad'], expected['projection_grad'])
        backend_cosine = cosine(actual['backend_grad'], expected['backend_grad'])
        cosine_pass = all(
            item['cosine'] is None or item['cosine'] >= GRAD_COSINE_MIN
            for item in (projection_cosine, backend_cosine)
        )
        item_pass = loss_error['pass'] and projection_error['pass'] and backend_error['pass'] and cosine_pass
        passed = passed and item_pass
        records.append({
            'repeat': repeat,
            'loss': float(actual['loss'].cpu()),
            'loss_error': loss_error,
            'projection_gradient_error': projection_error,
            'backend_gradient_error': backend_error,
            'projection_gradient': projection_cosine,
            'backend_gradient': backend_cosine,
            'positive_anchors': actual['positive_anchors'],
            'pass': item_pass,
        })
    return {
        'repeats': REPEATS,
        'reference_loss': float(expected['loss'].cpu()),
        'reference_loss_components': expected['loss_components'],
        'reference_positive_anchors': expected['positive_anchors'],
        'loss_distribution': sample_stats([item['loss'] for item in records]),
        'projection_gradient_norm_distribution': sample_stats([
            item['projection_gradient']['actual_norm'] for item in records
        ]),
        'backend_gradient_norm_distribution': sample_stats([
            item['backend_gradient']['actual_norm'] for item in records
        ]),
        'minimum_projection_cosine': min(
            item['projection_gradient']['cosine'] for item in records
            if item['projection_gradient']['cosine'] is not None
        ),
        'minimum_backend_cosine': min(
            item['backend_gradient']['cosine'] for item in records
            if item['backend_gradient']['cosine'] is not None
        ),
        'worst_loss_abs_error': max(item['loss_error']['max_abs'] for item in records),
        'worst_projection_grad_abs_error': max(item['projection_gradient_error']['max_abs'] for item in records),
        'worst_backend_grad_abs_error': max(item['backend_gradient_error']['max_abs'] for item in records),
        'records': records,
        'pass': passed,
    }


@torch.no_grad()
def amp_audit(context):
    restore_models(context, train=False)
    batch = context['validation_batch']
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        ref = context['reference'].vfe(clone_batch(batch))['pillar_features']
        opt = context['optimized'].vfe(clone_batch(batch))['pillar_features']
    result = tensor_error(opt, ref, HIST_RTOL, HIST_ATOL)
    result.update({'reference_dtype': str(ref.dtype), 'optimized_dtype': str(opt.dtype)})
    result['pass'] = result['pass'] and ref.dtype == opt.dtype == torch.float32
    return result


def event_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), output


@torch.no_grad()
def memory_increment(fn):
    torch.cuda.synchronize()
    baseline = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())
    del output
    return {'baseline_bytes': baseline, 'peak_bytes': peak, 'increment_bytes': peak - baseline}


@torch.no_grad()
def benchmark(context, rounds=5, vfe_iterations=200, e2e_iterations=100, warmup=30):
    restore_models(context, train=False)
    batch = context['validation_batch']
    ref = context['reference']
    opt = context['optimized']
    fns = {
        'vfe': {
            'ref': lambda: ref.vfe(clone_batch(batch)),
            'opt': lambda: opt.vfe(clone_batch(batch)),
        },
        'end_to_end': {
            'ref': lambda: ref(clone_batch(batch)),
            'opt': lambda: opt(clone_batch(batch)),
        },
    }
    results = {'protocol': {
        'timer': 'torch.cuda.Event with end.synchronize per sample',
        'warmup': warmup,
        'vfe_iterations_per_implementation_per_round': vfe_iterations,
        'end_to_end_iterations_per_implementation_per_round': e2e_iterations,
        'rounds': rounds,
        'interleave': 'REF/OPT order alternates by sample and round',
        'paired_ci': 'mean(REF_ms-OPT_ms) +/- 1.96*sample_standard_error over all paired samples',
    }, 'rounds': []}

    for name in fns:
        for implementation in ('ref', 'opt'):
            for _ in range(warmup):
                fns[name][implementation]()
        torch.cuda.synchronize()

    for round_index in range(rounds):
        round_result = {'round': round_index, 'measurements': {}}
        for scope, iterations in (('vfe', vfe_iterations), ('end_to_end', e2e_iterations)):
            samples = {'ref': [], 'opt': []}
            for sample_index in range(iterations):
                order = ('ref', 'opt') if (round_index + sample_index) % 2 == 0 else ('opt', 'ref')
                measured = {}
                for implementation in order:
                    elapsed, output = event_ms(fns[scope][implementation])
                    measured[implementation] = elapsed
                    del output
                samples['ref'].append(measured['ref'])
                samples['opt'].append(measured['opt'])
            ref_stats = sample_stats(samples['ref'])
            opt_stats = sample_stats(samples['opt'])
            round_result['measurements'][scope] = {
                'ref_ms': ref_stats,
                'opt_ms': opt_stats,
                'p50_speedup_percent': 100.0 * (ref_stats['p50'] - opt_stats['p50']) / ref_stats['p50'],
                'paired_difference_ms': [r - o for r, o in zip(samples['ref'], samples['opt'])],
                'raw_ref_ms': samples['ref'],
                'raw_opt_ms': samples['opt'],
            }
        results['rounds'].append(round_result)

    results['memory'] = {
        scope: {
            implementation: memory_increment(fns[scope][implementation])
            for implementation in ('ref', 'opt')
        }
        for scope in ('vfe', 'end_to_end')
    }
    summary = {}
    for scope in ('vfe', 'end_to_end'):
        ref_values = []
        opt_values = []
        paired = []
        round_speedups = []
        for item in results['rounds']:
            measurement = item['measurements'][scope]
            ref_values.extend(measurement['raw_ref_ms'])
            opt_values.extend(measurement['raw_opt_ms'])
            paired.extend(measurement['paired_difference_ms'])
            round_speedups.append(measurement['p50_speedup_percent'])
        paired_mean = statistics.fmean(paired)
        paired_se = statistics.stdev(paired) / math.sqrt(len(paired))
        memory = results['memory'][scope]
        ref_memory = memory['ref']['increment_bytes']
        opt_memory = memory['opt']['increment_bytes']
        summary[scope] = {
            'ref_ms': sample_stats(ref_values),
            'opt_ms': sample_stats(opt_values),
            'aggregate_p50_speedup_percent': 100.0 * (percentile(ref_values, 0.5) - percentile(opt_values, 0.5)) / percentile(ref_values, 0.5),
            'positive_p50_rounds': sum(value > 0 for value in round_speedups),
            'round_p50_speedup_percent': round_speedups,
            'paired_mean_difference_ms': paired_mean,
            'paired_95ci_ms': [paired_mean - 1.96 * paired_se, paired_mean + 1.96 * paired_se],
            'memory_increment_percent': 100.0 * (opt_memory - ref_memory) / ref_memory if ref_memory else None,
            'throughput_ref_samples_per_second': 2000.0 / percentile(ref_values, 0.5),
            'throughput_opt_samples_per_second': 2000.0 / percentile(opt_values, 0.5),
        }
    results['summary'] = summary
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('formal R6 evaluation requires CUDA')
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    set_mode(args.seed, deterministic_algorithms=False)
    context = build_context(args.seed)

    manifest = {
        'schema_version': 'pillarhist-r6-opt-evaluation-v1',
        'status': 'FORMAL',
        'started_utc': utc_now(),
        'seed': args.seed,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest(),
        'source_sha256': {
            'pillar_hist_vfe.py': sha256(Path(__file__).resolve().parents[2] / 'pcdet/models/backbones_3d/vfe/pillar_hist_vfe.py'),
            'test_optimized.py': sha256(Path(__file__).resolve().parents[2] / 'tests/pillarhist/test_optimized.py'),
            'evaluate_pillarhist_opt.py': sha256(__file__),
        },
        'environment': {
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'amp': False,
        },
        'checkpoint': context['checkpoint'],
        'tolerances': {
            'hist_feature_loss': {'rtol': HIST_RTOL, 'atol': HIST_ATOL},
            'gradient': {'rtol': GRAD_RTOL, 'atol': GRAD_ATOL, 'cosine_min': GRAD_COSINE_MIN},
        },
    }
    (args.output_root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    correctness = {
        'validation_eval': compare_eval(context, 'validation', context['validation_batch']),
        'training_fixture_eval': compare_eval(context, 'training', context['training_batch']),
        'outer_amp': amp_audit(context),
        'atomic_repeatability_loss_gradient': repeatability_audit(context),
    }
    correctness['pass'] = all([
        correctness['validation_eval']['pass'],
        correctness['training_fixture_eval']['pass'],
        correctness['outer_amp']['pass'],
        correctness['atomic_repeatability_loss_gradient']['pass'],
    ])
    correctness_path = args.output_root / 'correctness' / 'tst018.json'
    correctness_path.parent.mkdir()
    correctness_path.write_text(json.dumps(correctness, indent=2), encoding='utf-8')
    if not correctness['pass']:
        decision = {'status': 'BLOCKED/FAIL', 'reason': 'TST-018 correctness failure'}
    else:
        performance = benchmark(context)
        benchmark_path = args.output_root / 'benchmark' / 'benchmark.json'
        benchmark_path.parent.mkdir()
        benchmark_path.write_text(json.dumps(performance, indent=2), encoding='utf-8')
        vfe = performance['summary']['vfe']
        e2e = performance['summary']['end_to_end']
        vfe_gate = vfe['aggregate_p50_speedup_percent'] >= 10.0 and vfe['positive_p50_rounds'] >= 4
        e2e_gate = e2e['paired_95ci_ms'][0] > 0.0
        memory_gate = all(
            performance['summary'][scope]['memory_increment_percent'] is not None
            and performance['summary'][scope]['memory_increment_percent'] <= 10.0
            for scope in ('vfe', 'end_to_end')
        )
        if vfe_gate and e2e_gate and memory_gate:
            status = 'PASS_PROMOTED'
        elif vfe_gate:
            status = 'PASS_LOCAL_ONLY'
        else:
            status = 'PASS_NOT_PROMOTED'
        decision = {
            'status': status,
            'tst018': 'PASS',
            'selected_r7_reduction': 'compact_lookup_segment' if status == 'PASS_PROMOTED' else 'deterministic_segment',
            'gates': {
                'vfe_p50_at_least_10_percent_and_4_of_5_positive': vfe_gate,
                'end_to_end_paired_95ci_lower_above_zero': e2e_gate,
                'peak_memory_increase_at_most_10_percent': memory_gate,
            },
            'key_metrics': {
                'vfe_p50_speedup_percent': vfe['aggregate_p50_speedup_percent'],
                'vfe_positive_rounds': vfe['positive_p50_rounds'],
                'end_to_end_p50_speedup_percent': e2e['aggregate_p50_speedup_percent'],
                'end_to_end_paired_95ci_ms': e2e['paired_95ci_ms'],
                'vfe_memory_increase_percent': vfe['memory_increment_percent'],
                'end_to_end_memory_increase_percent': e2e['memory_increment_percent'],
            },
        }
    decision['finished_utc'] = utc_now()
    (args.output_root / 'decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    print(json.dumps(decision, sort_keys=True))


if __name__ == '__main__':
    main()
