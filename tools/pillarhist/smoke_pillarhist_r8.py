#!/usr/bin/env python3
"""Exploratory batch/memory/recovery gate used before freezing the R8 preregistration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from r8_protocol import (
    HASH_CHAIN_INITIAL,
    TRACK_SPECS,
    apply_canonical_initialization,
    build_dataset,
    configure_determinism,
    configure_track,
    environment_record,
    json_dump,
    make_canonical_package,
    make_replay_entries,
    sha256_file,
    tensor_dict_checksum,
    update_hash_chain,
    utc_now,
)
from run_pillarhist_r8_track import (
    load_checkpoint,
    recovery_continuity_check,
    run_update,
    save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    repo = Path(__file__).resolve().parents[2]
    determinism = configure_determinism(args.seed)
    configs = {
        track: configure_track(repo, track, args.batch_size, 80)
        for track in TRACK_SPECS
    }
    canonical_dataset, _ = build_dataset(
        configs["PH_PAPER_LITERAL_FULL"], True, args.seed, args.batch_size
    )
    entries = list(make_replay_entries(
        len(canonical_dataset), args.batch_size, 1, args.seed
    ))[:args.steps]

    from pcdet.models import build_network
    from train_utils.optimization import build_optimizer, build_scheduler

    torch.manual_seed(args.seed)
    canonical_model = build_network(
        configs["PH_PAPER_LITERAL_FULL"].MODEL,
        len(configs["PH_PAPER_LITERAL_FULL"].CLASS_NAMES),
        canonical_dataset,
    )
    package = make_canonical_package(canonical_model, args.seed)
    canonical_path = args.output_root / "canonical_initialization.pth"
    torch.save(package, canonical_path)
    canonical_sha256 = sha256_file(canonical_path)
    replay_sha256 = tensor_dict_checksum({"placeholder": torch.tensor([args.seed, args.batch_size])})

    track_results = {}
    for track, config in configs.items():
        dataset, loader = build_dataset(config, True, args.seed, args.batch_size)
        torch.manual_seed(args.seed)
        model = build_network(config.MODEL, len(config.CLASS_NAMES), dataset)
        initialization = apply_canonical_initialization(model, package, track)
        optimizer = build_optimizer(model, config.OPTIMIZATION)
        scheduler, warmup = build_scheduler(
            optimizer,
            total_iters_each_epoch=len(loader),
            total_epochs=80,
            last_epoch=-1,
            optim_cfg=config.OPTIMIZATION,
        )
        if warmup is not None:
            raise AssertionError("unexpected LR warmup")
        model.cuda().train()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        chain = HASH_CHAIN_INITIAL
        records = []
        recovery = None
        step_seconds = []
        checkpoint_path = args.output_root / track / "smoke_checkpoint.pth"
        for index, entry in enumerate(entries):
            if index == 2:
                save_checkpoint(
                    checkpoint_path, model, optimizer, dataset, track, args.seed,
                    0, index, replay_sha256, canonical_sha256, chain, {}, len(loader) * 80,
                )
                recovery = recovery_continuity_check(
                    checkpoint_path, model, optimizer, scheduler, dataset, entry,
                    config, replay_sha256, canonical_sha256,
                )
            step_started = time.time()
            result = run_update(
                model, optimizer, scheduler, dataset, entry, config,
                collect_gradient=(index == 0 or (index + 1) % 100 == 0),
            )
            step_seconds.append(time.time() - step_started)
            chain = update_hash_chain(
                chain, entry["global_step_zero_based"], result["checksums"]["input"]
            )
            records.append({
                "step": index + 1,
                "loss": result["loss"],
                "input_checksum": result["checksums"]["input"],
                "gradient": result["gradient"],
            })
        elapsed = time.time() - started
        measured_step_seconds = step_seconds[min(5, len(step_seconds) - 1):]
        track_results[track] = {
            "initialization": initialization,
            "records": records,
            "input_hash_chain_final_root": chain,
            "recovery_continuity": recovery,
            "elapsed_seconds": elapsed,
            "steps_per_second": len(measured_step_seconds) / sum(measured_step_seconds),
            "step_seconds_after_warmup": measured_step_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "steps_per_epoch": len(loader),
            "train_frames": len(dataset),
        }
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    pp = track_results["PP_OFFICIAL_FULL"]
    ph = track_results["PH_PAPER_LITERAL_FULL"]
    checksums_equal = [record["input_checksum"] for record in pp["records"]] == [
        record["input_checksum"] for record in ph["records"]
    ]
    pass_gate = all([
        checksums_equal,
        pp["input_hash_chain_final_root"] == ph["input_hash_chain_final_root"],
        pp["recovery_continuity"] is not None and pp["recovery_continuity"]["pass"],
        ph["recovery_continuity"] is not None and ph["recovery_continuity"]["pass"],
        all(item["gradient"]["pass"] for item in pp["records"] if item["gradient"] is not None),
        all(item["gradient"]["pass"] for item in ph["records"] if item["gradient"] is not None),
    ])
    payload = {
        "schema_version": "pillarhist-r8-exploratory-smoke-v1",
        "status": "PASS" if pass_gate else "FAIL",
        "exploratory": True,
        "created_utc": utc_now(),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "workers": 0,
        "epochs_for_scheduler": 80,
        "steps_executed": args.steps,
        "determinism": determinism,
        "environment": environment_record(repo),
        "track_results": track_results,
        "paired_input_checksums_equal": checksums_equal,
        "paired_hash_chain_equal": pp["input_hash_chain_final_root"] == ph["input_hash_chain_final_root"],
    }
    json_dump(args.output_root / "smoke_result.json", payload)
    print(json.dumps(payload, indent=2))
    if not pass_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
