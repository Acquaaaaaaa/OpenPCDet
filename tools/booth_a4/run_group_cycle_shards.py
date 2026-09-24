"""Run resumable PointPillars group/cycle profiling shards and merge them."""

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
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--correction-parallelism", type=int, default=8)
    parser.add_argument("--scalar-correction-parallelism", type=int, default=8)
    parser.add_argument("--main-output-parallelism", type=int)
    parser.add_argument("--correction-output-parallelism", type=int)
    parser.add_argument("--padding-policy", default="physical_rows_included")
    parser.add_argument("--pfn-token-policy", default="fixed_slots_included")
    parser.add_argument("--chunk-streams", type=int, default=16384)
    parser.add_argument("--mapping-mode", default="consumer_aware_reference_cim")
    parser.add_argument("--conv-transpose-mapping", default="direct_scatter")
    parser.add_argument("--scale-methods", nargs="+", default=("minmax", "p99_99", "p99_9"))
    parser.add_argument("--metrics-mode", choices=("full", "core"), default="full")
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
            checks = json.loads(
                (destination / "validation_checks.json").read_text(encoding="utf-8")
            )
            if checks.get("status") != "complete" or checks.get("processed_frame_ids") != expected_ids:
                raise ValueError(f"existing shard failed resume validation: {destination}")
            completed += 1
            continue
        if args.max_shards is not None and launched >= args.max_shards:
            break
        partial = shards_dir / f".{shard_name}.partial"
        if partial.exists():
            raise ValueError(f"partial shard requires inspection before retry: {partial}")
        command = [
            sys.executable, "-m", "tools.booth_a4.profile_group_cycle_booth",
            "--cfg-file", str(args.cfg_file.resolve()),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--manifest", str(args.manifest.resolve()),
            "--scales", str(args.scales.resolve()),
            "--output-dir", str(partial.resolve()),
            "--start-index", str(start),
            "--max-frames", str(end - start),
            "--workers", str(args.workers),
            "--group-size", str(args.group_size),
            "--correction-parallelism", str(args.correction_parallelism),
            "--scalar-correction-parallelism", str(args.scalar_correction_parallelism),
            "--padding-policy", args.padding_policy,
            "--pfn-token-policy", args.pfn_token_policy,
            "--chunk-streams", str(args.chunk_streams),
            "--mapping-mode", args.mapping_mode,
            "--conv-transpose-mapping", args.conv_transpose_mapping,
            "--scale-methods", *args.scale_methods,
            "--metrics-mode", args.metrics_mode,
        ]
        if args.main_output_parallelism is not None:
            command.extend(("--main-output-parallelism", str(args.main_output_parallelism)))
        if args.correction_output_parallelism is not None:
            command.extend((
                "--correction-output-parallelism", str(args.correction_output_parallelism)
            ))
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
        command = [
            sys.executable, "-m", "tools.booth_a4.merge_group_cycle_shards",
            "--shards-dir", str(shards_dir.resolve()),
            "--manifest", str(args.manifest.resolve()),
            "--output-dir", str(merged.resolve()),
            "--scalar-correction-parallelism", str(args.scalar_correction_parallelism),
            "--metrics-mode", args.metrics_mode,
        ]
        if args.main_output_parallelism is not None:
            command.extend(("--main-output-parallelism", str(args.main_output_parallelism)))
        if args.correction_output_parallelism is not None:
            command.extend((
                "--correction-output-parallelism", str(args.correction_output_parallelism)
            ))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
