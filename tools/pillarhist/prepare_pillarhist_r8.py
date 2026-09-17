#!/usr/bin/env python3
"""Prepare one R8 seed: shared replay, canonical backend, and frozen manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from r8_protocol import (
    SCHEMA_VERSION,
    TRACK_SPECS,
    apply_canonical_initialization,
    build_dataset,
    configure_determinism,
    configure_track,
    environment_record,
    json_dump,
    make_canonical_package,
    sha256_file,
    to_plain,
    utc_now,
    write_replay,
)


def git_diff_sha256(repo: Path) -> str:
    return hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"], cwd=repo)).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(666, 667, 668), required=True)
    parser.add_argument("--batch-size", type=int, choices=(2, 4), default=4)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    if args.epochs != 80:
        raise ValueError("formal R8 preparation requires exactly 80 epochs")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)

    repo = Path(__file__).resolve().parents[2]
    determinism = configure_determinism(args.seed)
    configs = {
        track: configure_track(repo, track, args.batch_size, args.epochs)
        for track in TRACK_SPECS
    }
    train_dataset, train_loader = build_dataset(
        configs["PH_PAPER_LITERAL_FULL"], True, args.seed, args.batch_size
    )
    validation_dataset, _ = build_dataset(
        configs["PH_PAPER_LITERAL_FULL"], False, args.seed, args.batch_size
    )
    dataset_size = len(train_dataset)
    validation_size = len(validation_dataset)
    if dataset_size != 3712 or validation_size != 3769:
        raise AssertionError({
            "expected_train": 3712,
            "actual_train": dataset_size,
            "expected_validation": 3769,
            "actual_validation": validation_size,
        })
    if len(train_loader) != 928 if args.batch_size == 4 else len(train_loader) != 1856:
        raise AssertionError("unexpected steps per epoch")

    replay_path = args.output_root / "replay_steps.jsonl.gz"
    replay_manifest = write_replay(
        replay_path, dataset_size, args.batch_size, args.epochs, args.seed
    )
    json_dump(args.output_root / "replay_manifest.json", replay_manifest)

    from pcdet.models import build_network

    torch.manual_seed(args.seed)
    canonical_model = build_network(
        configs["PH_PAPER_LITERAL_FULL"].MODEL,
        len(configs["PH_PAPER_LITERAL_FULL"].CLASS_NAMES),
        train_dataset,
    )
    package = make_canonical_package(canonical_model, args.seed)
    canonical_path = args.output_root / "canonical_initialization.pth"
    torch.save(package, canonical_path)
    initialization = {}
    for track, config in configs.items():
        torch.manual_seed(args.seed)
        model = build_network(config.MODEL, len(config.CLASS_NAMES), train_dataset)
        initialization[track] = apply_canonical_initialization(model, package, track)
        del model
    init_manifest = {
        "schema_version": f"{SCHEMA_VERSION}-canonical-init-manifest",
        "created_utc": utc_now(),
        "seed": args.seed,
        "optimizer_created_after_copy": True,
        "backend_checksum": package["backend_checksum"],
        "canonical_file": str(canonical_path),
        "canonical_file_sha256": sha256_file(canonical_path),
        "tracks": initialization,
    }
    json_dump(args.output_root / "canonical_initialization.json", init_manifest)

    resolved_path = args.output_root / "resolved_track_configs.yaml"
    resolved_path.write_text(
        yaml.safe_dump({track: to_plain(config) for track, config in configs.items()}, sort_keys=False),
        encoding="utf-8",
    )
    environment = environment_record(repo)
    json_dump(args.output_root / "environment.json", environment)
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}-preparation",
        "status": "PREPARED",
        "created_utc": utc_now(),
        "seed": args.seed,
        "tracks": list(TRACK_SPECS),
        "batch_size": args.batch_size,
        "global_batch_size": args.batch_size,
        "gradient_accumulation": 1,
        "workers": 0,
        "epochs": args.epochs,
        "steps_per_epoch": replay_manifest["steps_per_epoch"],
        "total_optimizer_steps": replay_manifest["total_optimizer_steps"],
        "drop_last": False,
        "train_frames": dataset_size,
        "validation_frames": validation_size,
        "determinism": determinism,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        "git_diff_sha256": git_diff_sha256(repo),
        "replay_manifest": {
            "path": str(args.output_root / "replay_manifest.json"),
            "sha256": sha256_file(args.output_root / "replay_manifest.json"),
        },
        "canonical_initialization": {
            "path": str(canonical_path),
            "sha256": sha256_file(canonical_path),
        },
        "resolved_configs": {"path": str(resolved_path), "sha256": sha256_file(resolved_path)},
        "source_sha256": {
            path.name: sha256_file(path)
            for path in [
                Path(__file__),
                Path(__file__).with_name("r8_protocol.py"),
                Path(__file__).with_name("run_pillarhist_r8_track.py"),
                Path(__file__).with_name("audit_pillarhist_r8.py"),
            ]
        },
    }
    json_dump(args.output_root / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
