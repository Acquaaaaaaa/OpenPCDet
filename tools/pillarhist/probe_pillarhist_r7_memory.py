#!/usr/bin/env python3
"""Measure one fixed post-training backward pass for R7 memory accounting."""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch


def json_dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the R7 memory probe")

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    import run_pillarhist_r7_track as runner

    replay_path = args.preparation_root / "training_replay_manifest.json"
    canonical_path = args.preparation_root / "canonical_initialization.pth"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    package = torch.load(canonical_path, map_location="cpu", weights_only=False)
    replay_sha256 = runner.sha256_file(replay_path)
    canonical_sha256 = runner.sha256_file(canonical_path)

    repo = script_dir.parents[1]
    ph_config = runner.load_config(repo / "tools/cfgs/kitti_models/pointpillar_pillarhist.yaml")
    pp_config = runner.load_config(repo / "tools/cfgs/kitti_models/pointpillar.yaml")
    config = runner.configure_track(ph_config, pp_config, args.track)
    seed = int(replay["seed"])
    dataset = runner.build_dataset(ph_config, training=True, seed=seed)

    from pcdet.models import build_network, load_data_to_gpu

    runner.set_seed(seed, cuda=True)
    model = build_network(config.MODEL, len(config.CLASS_NAMES), dataset)
    runner.apply_canonical_initialization(model, package, args.track)
    optimizer, _ = runner.make_optimizer_and_scheduler(model, config.OPTIMIZATION, 5000)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model.cuda().train()
    checkpoint_path = args.track_root / "checkpoints/last.pth"
    checkpoint = runner.load_checkpoint(
        checkpoint_path, model, optimizer, scaler, dataset,
        replay_sha256, canonical_sha256,
    )
    probe_entry = replay["entries"][int(checkpoint["completed_optimizer_steps"])]
    batch, checksums = runner.verify_replay_entry(dataset, probe_entry)
    load_data_to_gpu(batch)

    optimizer.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    result, tb, _ = model(batch)
    loss = result["loss"]
    positive_anchors = int((model.dense_head.forward_ret_dict["box_cls_labels"] > 0).sum())
    loss.backward()
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    payload = {
        "schema_version": "pillarhist-r7-memory-probe-v1",
        "track": args.track,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_completed_optimizer_steps": int(checkpoint["completed_optimizer_steps"]),
        "probe_step_zero_based": int(probe_entry["step_zero_based"]),
        "probe_input_checksum": checksums["input"],
        "loss": float(loss.detach().cpu()),
        "loss_components": {key: float(value) for key, value in tb.items()},
        "positive_anchors": positive_anchors,
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_allocated_bytes": peak_allocated - baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_reserved_bytes": peak_reserved - baseline_reserved,
        "scope": "single fixed resume-probe forward/backward; diagnostic only",
    }
    json_dump(args.output, payload)
    print(json.dumps(payload, sort_keys=True))

    del batch, loss, result, model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
