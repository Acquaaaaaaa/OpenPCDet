"""Benchmark conservative DataLoader settings with exact count equivalence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from .compare_profile_runs import compare_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    run_dirs = {}
    for workers in (0, 2):
        run_dir = args.output_dir / f"workers_{workers}"
        command = [
            sys.executable, "-m", "tools.booth_a4.profile_activation_booth_a4",
            "--cfg-file", str(args.cfg_file.resolve()),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--manifest", str(args.manifest.resolve()),
            "--scales", str(args.scales.resolve()),
            "--output-dir", str(run_dir.resolve()),
            "--max-frames", "32",
            "--workers", str(workers),
        ]
        subprocess.run(command, check=True)
        checks = json.loads((run_dir / "validation_checks.json").read_text(encoding="utf-8"))
        rows.append({
            "batch_size": 1,
            "workers": workers,
            "frame_count": checks["processed_frame_count"],
            "elapsed_seconds": checks["elapsed_seconds"],
            "frames_per_second": checks["processed_frame_count"] / checks["elapsed_seconds"],
        })
        run_dirs[workers] = run_dir

    equivalence = compare_runs(run_dirs[0], run_dirs[2])
    if not equivalence["all_integer_results_exactly_equal"]:
        raise RuntimeError("workers candidate changed integer profiling results")
    recommended = min(rows, key=lambda row: row["elapsed_seconds"])
    for row in rows:
        row["integer_counts_equal_to_baseline"] = True
        row["selected"] = row["workers"] == recommended["workers"]
    with (args.output_dir / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "complete",
        "frame_count": 32,
        "batch_size": 1,
        "batch_size_candidate_status": "not_tested",
        "batch_size_candidate_reason": (
            "PFN inputs concatenate pillars across a batch; exact per-frame attribution is not "
            "implemented, so the formal workflow remains batch_size=1"
        ),
        "workers_tested": [0, 2],
        "recommended_workers": recommended["workers"],
        "integer_results_exactly_equal": True,
        "comparison": equivalence,
    }
    (args.output_dir / "validation_checks.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
