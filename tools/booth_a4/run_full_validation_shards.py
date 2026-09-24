"""Run resumable 256-frame full-validation shards and merge them strictly."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .build_frame_lists import read_frame_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    frame_ids = read_frame_ids(args.manifest)
    shards_dir = args.output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    launched = 0
    for start in range(0, len(frame_ids), args.shard_size):
        end = min(start + args.shard_size, len(frame_ids))
        shard_name = f"{start:04d}-{end - 1:04d}"
        destination = shards_dir / shard_name
        expected_ids = frame_ids[start:end]
        if destination.exists():
            checks_path = destination / "validation_checks.json"
            if not checks_path.exists():
                raise ValueError(f"existing shard has no validation checks: {destination}")
            checks = json.loads(checks_path.read_text(encoding="utf-8"))
            if checks.get("status") != "complete" or checks.get("processed_frame_ids") != expected_ids:
                raise ValueError(f"existing shard failed resume validation: {destination}")
            print(f"skip validated shard {shard_name}", flush=True)
            completed += 1
            continue
        if args.max_shards is not None and launched >= args.max_shards:
            break

        partial = shards_dir / f".{shard_name}.partial"
        if partial.exists():
            raise ValueError(
                f"partial shard exists and requires inspection before retry: {partial}"
            )
        command = [
            sys.executable, "-m", "tools.booth_a4.profile_activation_booth_a4",
            "--cfg-file", str(args.cfg_file.resolve()),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--manifest", str(args.manifest.resolve()),
            "--scales", str(args.scales.resolve()),
            "--output-dir", str(partial.resolve()),
            "--start-index", str(start),
            "--max-frames", str(end - start),
            "--workers", str(args.workers),
        ]
        print(f"run shard {shard_name}", flush=True)
        subprocess.run(command, check=True)
        checks = json.loads((partial / "validation_checks.json").read_text(encoding="utf-8"))
        if checks.get("status") != "complete" or checks.get("processed_frame_ids") != expected_ids:
            raise RuntimeError(f"new shard failed completion validation: {shard_name}")
        os.replace(partial, destination)
        completed += 1
        launched += 1

    total_shards = (len(frame_ids) + args.shard_size - 1) // args.shard_size
    status = {
        "total_shards": total_shards,
        "completed_shards": completed,
        "launched_this_run": launched,
        "all_shards_complete": completed == total_shards,
    }
    (args.output_dir / "shard_progress.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status), flush=True)
    if completed == total_shards:
        merged = args.output_dir / "merged"
        if merged.exists():
            raise FileExistsError(f"merged output already exists: {merged}")
        subprocess.run(
            [
                sys.executable, "-m", "tools.booth_a4.merge_booth_a4_shards",
                "--shards-dir", str(shards_dir.resolve()),
                "--manifest", str(args.manifest.resolve()),
                "--scales", str(args.scales.resolve()),
                "--output-dir", str(merged.resolve()),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
