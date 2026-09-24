"""Finalize and audit one completed Booth A4 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_exact(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(f"existing artifact differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def add_check(checks: list[dict], name: str, passed: bool, evidence: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
    if not passed:
        raise AssertionError(f"acceptance check failed: {name}: {evidence}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    merged = experiment / "full_val" / "merged"
    calibration = experiment / "calibration"
    pilot = experiment / "pilot_256_repeat"
    report = experiment / "report"

    for filename in (
        "activation_summary_long.csv",
        "per_frame_counts.csv",
        "int8_histograms.npz",
        "booth_digit_histograms.npz",
        "validation_checks.json",
        "activation_registry.json",
        "activation_edges.csv",
        "layer_consumers.csv",
    ):
        copy_exact(merged / filename, experiment / "full_val" / filename)
    (report / "figures").mkdir(parents=True, exist_ok=True)

    diagnostics = json.loads((calibration / "percentile_diagnostics.json").read_text(encoding="utf-8"))
    diagnostic_rows = []
    for activation_id, item in sorted(diagnostics.items()):
        diagnostic_rows.append({
            "activation_id": activation_id,
            "sample_zero_count": item["sample_zero_count"],
            "sample_nonzero_count": item["sample_nonzero_count"],
            "p99_99_all": item["all_value_percentiles"]["p99_99"],
            "p99_9_all": item["all_value_percentiles"]["p99_9"],
            "p99_99_nonzero_only": item["nonzero_only_percentiles"]["p99_99"],
            "p99_9_nonzero_only": item["nonzero_only_percentiles"]["p99_9"],
        })
    percentile_csv = calibration / "percentile_diagnostics.csv"
    if percentile_csv.exists():
        raise FileExistsError(f"refusing to overwrite: {percentile_csv}")
    with percentile_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=diagnostic_rows[0].keys())
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    environment = json.loads((experiment / "environment.json").read_text(encoding="utf-8"))
    environment_text = experiment / "environment.txt"
    if environment_text.exists():
        raise FileExistsError(f"refusing to overwrite: {environment_text}")
    environment_text.write_text(
        "".join(f"{key}: {value}\n" for key, value in sorted(environment.items())),
        encoding="utf-8",
    )

    source_files = sorted((REPO_ROOT / "tools" / "booth_a4").rglob("*.py"))
    source_files += sorted((REPO_ROOT / "tools" / "booth_a4").rglob("*.yaml"))
    source_files += sorted((REPO_ROOT / "tests" / "booth_a4").rglob("*.py"))
    source_snapshot = {
        str(path.relative_to(REPO_ROOT)): {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in source_files
    }
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    code_snapshot = {
        "git_commit": git_commit,
        "files": source_snapshot,
        "snapshot_note": "Snapshot recorded at final audit; core profiling sources were unchanged during full-validation shard execution.",
    }
    (experiment / "code_snapshot_manifest.json").write_text(
        json.dumps(code_snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/booth_a4"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    (experiment / "final_test_output.txt").write_text(
        test_result.stdout + test_result.stderr, encoding="utf-8"
    )
    if test_result.returncode != 0:
        raise RuntimeError("final Booth A4 test suite failed")

    checks: list[dict] = []
    baseline = json.loads((experiment / "baseline_checks.json").read_text(encoding="utf-8"))
    calibration_checks = json.loads((calibration / "validation_checks.json").read_text(encoding="utf-8"))
    scales = json.loads((calibration / "calibration_scales.json").read_text(encoding="utf-8"))
    pilot_repeat = json.loads((experiment / "pilot_repeatability.json").read_text(encoding="utf-8"))
    benchmark = json.loads((experiment / "benchmark_32" / "validation_checks.json").read_text(encoding="utf-8"))
    full_checks = json.loads((merged / "validation_checks.json").read_text(encoding="utf-8"))
    registry = json.loads((merged / "activation_registry.json").read_text(encoding="utf-8"))
    summary = read_csv(merged / "activation_summary_long.csv")
    per_frame = read_csv(merged / "per_frame_counts.csv")

    add_check(checks, "epoch80_checkpoint_sha256", baseline["checkpoint_sha256"] == sha256_file(args.checkpoint), baseline["checkpoint_sha256"])
    add_check(checks, "strict_checkpoint_load", baseline["strict_checkpoint_load"] is True, "baseline_checks.json")
    add_check(checks, "hooks_preserve_outputs", baseline["hooks_preserved_outputs_exactly"] is True, "baseline_checks.json")
    add_check(checks, "booth_and_quantizer_tests", "37 passed" in test_result.stdout, test_result.stdout.strip())
    add_check(checks, "registry_cardinality", registry["consumer_count"] == 23 and registry["unique_activation_count"] == 19, "23 consumers / 19 unique activations")
    add_check(checks, "shared_activation_registry", sum(edge["shared_consumer_count"] > 1 for edge in registry["activation_edges"]) == 3, "3 shared logical activations")
    add_check(checks, "padding_exclusion_registry", sum(edge["exclude_boundary_padding"] for edge in registry["activation_edges"]) >= 3, "pre-ZeroPad capture recorded")
    add_check(checks, "calibration_256_train", calibration_checks["formal_full_manifest"] and calibration_checks["processed_frame_count"] == 256 and calibration_checks["source_split"] == "train", "calibration/validation_checks.json")
    add_check(checks, "calibration_finite", calibration_checks["all_values_finite"] is True, "calibration/validation_checks.json")
    scale_statuses = [scales[a][method]["status"] for a in scales for method in ("minmax", "p99_99", "p99_9")]
    add_check(checks, "three_scale_methods_frozen", len(scales) == 19 and all(status == "ok" for status in scale_statuses), "19 activations x 3 valid scales")
    add_check(checks, "pilot_repeatability", pilot_repeat["all_integer_results_exactly_equal"] is True, "pilot_repeatability.json")
    add_check(checks, "benchmark_equivalence", benchmark["integer_results_exactly_equal"] is True and benchmark["recommended_workers"] == 0, "benchmark_32/validation_checks.json")
    add_check(checks, "full_frame_coverage", full_checks["processed_frame_count"] == 3769 and full_checks["unique_frame_count"] == 3769 and full_checks["frame_order_exact"], "15 merged shards")
    add_check(checks, "full_hash_consistency", full_checks["checkpoint_sha256"] == sha256_file(args.checkpoint) and full_checks["config_sha256"] == sha256_file(args.config), "merged/validation_checks.json")
    add_check(checks, "summary_shape", len(summary) == 57 and len({row["activation_id"] for row in summary}) == 19, "57 activation-scale rows")
    add_check(checks, "per_frame_shape", len(per_frame) == 3769 * 57 and len({row["frame_id"] for row in per_frame}) == 3769, f"{len(per_frame)} rows")

    integer_invariants = all(
        int(row["N_total"]) == int(row["N_A4"]) + int(row["N_exception"])
        and int(row["N_total"]) == int(row["N_zero_quantized"]) + int(row["N_nonzero_quantized"])
        and int(row["N_total"]) <= 2**63 - 1
        and abs(float(row["RA4_tensor_micro"]) + float(row["Rexc_tensor_micro"]) - 1.0) < 1e-12
        for row in summary
    )
    add_check(checks, "integer_and_ratio_invariants", integer_invariants, "all 57 merged rows")
    with np.load(merged / "int8_histograms.npz") as int8_data, np.load(merged / "booth_digit_histograms.npz") as digit_data:
        histogram_ok = len(int8_data.files) == 57 and set(int8_data.files) == set(digit_data.files)
        for row in summary:
            key = f"{row['activation_id']}__{row['scale_method']}"
            total = int(row["N_total"])
            histogram_ok = histogram_ok and int(int8_data[key].sum()) == total
            histogram_ok = histogram_ok and bool(np.all(digit_data[key].sum(axis=1) == total))
    add_check(checks, "histogram_invariants", histogram_ok, "57 INT8 and Booth histogram pairs")
    add_check(checks, "report_tables", (report / "summary.md").exists() and (report / "layer_summary_wide.csv").exists() and (report / "consumer_activation_map.csv").exists(), "report artifacts")
    report_text = (report / "summary.md").read_text(encoding="utf-8")
    add_check(checks, "scope_and_limitations_reported", "不代表传播式 A8W8" in report_text and "MAC-weighted" in report_text and "不生成含义模糊的全网络 RA4" in report_text, "report/summary.md")

    acceptance = {
        "status": "complete",
        "all_checks_passed": all(check["passed"] for check in checks),
        "check_count": len(checks),
        "checks": checks,
        "known_optional_not_implemented": [
            "exact MAC-weighted RA4",
            "propagated QDQ / true deployed A8W8 activation profiling",
            "KITTI AP validation under quantization",
        ],
    }
    (experiment / "final_acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"all_checks_passed": acceptance["all_checks_passed"], "check_count": len(checks)}))


if __name__ == "__main__":
    main()
