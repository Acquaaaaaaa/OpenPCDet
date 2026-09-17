#!/usr/bin/env python3
"""Audit all six R8 runs and emit machine-readable closure artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from r8_protocol import (
    TRACK_SPECS,
    json_dump,
    json_load,
    recomputed_paper_values,
    scientific_label,
    sha256_file,
    utc_now,
)


SEEDS = (666, 667, 668)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def compare_training_logs(left, right):
    compared = 0
    for compared, (a, b) in enumerate(zip(read_jsonl(left), read_jsonl(right)), start=1):
        if a["optimizer_step_one_based"] != b["optimizer_step_one_based"]:
            return {"pass": False, "compared_steps": compared, "reason": "step mismatch"}
        if a["input_checksum"] != b["input_checksum"]:
            return {"pass": False, "compared_steps": compared, "reason": "input checksum mismatch"}
    left_count = sum(1 for _ in read_jsonl(left))
    right_count = sum(1 for _ in read_jsonl(right))
    return {
        "pass": left_count == right_count == compared,
        "compared_steps": compared,
        "left_steps": left_count,
        "right_steps": right_count,
        "reason": None if left_count == right_count == compared else "length mismatch",
    }


def compare_samples(left, right):
    left_values = list(read_jsonl_gz(left))
    right_values = list(read_jsonl_gz(right))
    return {
        "pass": left_values == right_values,
        "left_samples": len(left_values),
        "right_samples": len(right_values),
    }


def audit_track(root, expected_steps):
    required = [
        "manifest.json", "config_resolved.yaml", "environment.txt",
        "training_log.jsonl", "gradient_audit.json", "checkpoints/last.pth",
        "checkpoint_manifest.json", "validation/full/result.txt",
        "validation/full/metrics.json", "runtime_and_memory.json",
        "gpu_telemetry.csv", "replay_hash_chain.json", "track_summary.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return {"valid": False, "root": str(root), "missing": missing}
    manifest = json_load(root / "manifest.json")
    summary = json_load(root / "track_summary.json")
    gradient = json_load(root / "gradient_audit.json")
    chain = json_load(root / "replay_hash_chain.json")
    validation = json_load(root / "validation/full/metrics.json")
    checkpoint_manifest = json_load(root / "checkpoint_manifest.json")
    log_count = sum(1 for _ in read_jsonl(root / "training_log.jsonl"))
    retained_epochs = sorted({
        int(item["epoch"]) for item in checkpoint_manifest["checkpoints"]
        if "epoch" in item
    })
    finite_metrics = all(
        math.isfinite(float(value))
        for value in validation["metrics_raw_unrounded"].values()
    )
    valid = all([
        manifest["status"] == "COMPLETE",
        summary["status"] == "COMPLETE",
        summary["completed_epochs"] == 80,
        summary["completed_optimizer_steps"] == expected_steps,
        log_count == expected_steps,
        summary["all_losses_finite"],
        summary["all_positive_anchor_counts_above_zero"],
        summary["gradient_checks_pass"],
        gradient["pass"],
        bool(summary["resume_continuity"] and summary["resume_continuity"]["pass"]),
        chain["steps"] == expected_steps,
        chain["final_root"] == summary["input_hash_chain_final_root"],
        validation["frames"] == 3769,
        validation["prediction_count"] > 0,
        finite_metrics,
        all(epoch in retained_epochs for epoch in (20, 40, 60, 80)),
    ])
    return {
        "valid": valid,
        "root": str(root),
        "manifest": manifest,
        "summary": summary,
        "gradient_audit": gradient,
        "hash_chain": chain,
        "validation": validation,
        "checkpoint_manifest": checkpoint_manifest,
        "training_log_steps": log_count,
        "retained_checkpoint_epochs": retained_epochs,
        "last_checkpoint_sha256": sha256_file(root / "checkpoints/last.pth"),
    }


def metric_payload(audits):
    by_seed = {}
    for seed in SEEDS:
        by_seed[str(seed)] = {}
        for track in TRACK_SPECS:
            by_seed[str(seed)][track] = audits[seed][track]["validation"]["classes"]
    deltas = {}
    for seed in SEEDS:
        pp = by_seed[str(seed)]["PP_OFFICIAL_FULL"]
        ph = by_seed[str(seed)]["PH_PAPER_LITERAL_FULL"]
        deltas[str(seed)] = {
            category: {
                key: ph[category][key] - pp[category][key]
                for key in ("easy", "moderate", "hard", "mean_easy_moderate_hard")
            }
            for category in ("Car", "Pedestrian", "Cyclist")
        }
    aggregate = {}
    for track in TRACK_SPECS:
        aggregate[track] = {}
        for category in ("Car", "Pedestrian", "Cyclist"):
            aggregate[track][category] = {}
            for key in ("easy", "moderate", "hard", "mean_easy_moderate_hard"):
                values = [by_seed[str(seed)][track][category][key] for seed in SEEDS]
                aggregate[track][category][key] = {
                    "mean": statistics.fmean(values),
                    "sample_std": statistics.stdev(values),
                }
    delta_aggregate = {}
    for category in ("Car", "Pedestrian", "Cyclist"):
        delta_aggregate[category] = {}
        for key in ("easy", "moderate", "hard", "mean_easy_moderate_hard"):
            values = [deltas[str(seed)][category][key] for seed in SEEDS]
            delta_aggregate[category][key] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "positive_seed_count": sum(value > 0 for value in values),
            }
    return {
        "schema_version": "pillarhist-r8-metrics-v1",
        "numeric_source": "unrounded evaluator JSON",
        "by_seed": by_seed,
        "paired_deltas_by_seed": deltas,
        "aggregate": aggregate,
        "paired_delta_aggregate": delta_aggregate,
    }


def paper_comparison(metrics):
    paper = recomputed_paper_values()
    local = metrics["aggregate"]
    comparisons = {}
    mapping = {
        "PointPillars": "PP_OFFICIAL_FULL",
        "PH-PointPillars": "PH_PAPER_LITERAL_FULL",
    }
    for paper_name, track in mapping.items():
        comparisons[paper_name] = {}
        for category in ("Car", "Pedestrian"):
            comparisons[paper_name][category] = {}
            for key in ("easy", "moderate", "hard"):
                comparisons[paper_name][category][key] = {
                    "local_mean": local[track][category][key]["mean"],
                    "paper": paper[paper_name][category][key],
                    "local_minus_paper": local[track][category][key]["mean"] - paper[paper_name][category][key],
                }
            comparisons[paper_name][category]["map"] = {
                "local_mean": local[track][category]["mean_easy_moderate_hard"]["mean"],
                "paper_reported_map": paper[paper_name][category]["paper_reported_map"],
                "recomputed_from_displayed_ap": paper[paper_name][category]["recomputed_from_displayed_ap"],
            }
    paper_delta = {}
    for category in ("Car", "Pedestrian"):
        paper_delta[category] = {}
        for key in ("easy", "moderate", "hard"):
            expected = paper["PH-PointPillars"][category][key] - paper["PointPillars"][category][key]
            observed = metrics["paired_delta_aggregate"][category][key]["mean"]
            paper_delta[category][key] = {
                "paper_delta": expected,
                "local_paired_mean_delta": observed,
                "local_minus_paper_delta": observed - expected,
            }
    return {
        "schema_version": "pillarhist-r8-paper-comparison-v1",
        "paper_values": paper,
        "absolute_comparison": comparisons,
        "relative_improvement_comparison": paper_delta,
        "paper_protocol_mismatch": True,
        "protocol_items": {
            "gpu": "MISMATCH_OR_UNVERIFIED",
            "batch_size": "UNVERIFIED_AGAINST_PAPER",
            "learning_rate": "UNVERIFIED_AGAINST_PAPER",
            "openpcdet_version": "UNVERIFIED_AGAINST_PAPER",
            "data_files": "UNVERIFIED_AGAINST_PAPER",
            "training_and_evaluation_implementation": "MISMATCH_OR_UNVERIFIED",
            "pillar_center_interpretation": "LOCAL_LITERAL_INTERPRETATION_NOT_AUTHOR_CODE_CONFIRMED",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r8-root", type=Path, required=True)
    parser.add_argument("--docs-report", type=Path, required=True)
    args = parser.parse_args()
    audits = {}
    fairness_seeds = {}
    all_valid = True
    for seed in SEEDS:
        seed_root = args.r8_root / f"seed{seed}"
        prep = json_load(seed_root / "preparation_manifest.json")
        expected_steps = int(prep["total_optimizer_steps"])
        audits[seed] = {
            track: audit_track(seed_root / track, expected_steps)
            for track in TRACK_SPECS
        }
        tracks_valid = all(item["valid"] for item in audits[seed].values())
        if tracks_valid:
            pp = audits[seed]["PP_OFFICIAL_FULL"]
            ph = audits[seed]["PH_PAPER_LITERAL_FULL"]
            full_log = compare_training_logs(
                Path(pp["root"]) / "training_log.jsonl",
                Path(ph["root"]) / "training_log.jsonl",
            )
            samples = compare_samples(
                Path(pp["root"]) / "replay_samples.jsonl.gz",
                Path(ph["root"]) / "replay_samples.jsonl.gz",
            )
            roots_equal = (
                pp["hash_chain"]["epoch_roots"] == ph["hash_chain"]["epoch_roots"]
                and pp["hash_chain"]["final_root"] == ph["hash_chain"]["final_root"]
            )
        else:
            full_log = {"pass": False, "reason": "invalid track"}
            samples = {"pass": False, "reason": "invalid track"}
            roots_equal = False
        fairness_seeds[str(seed)] = {
            "tracks_valid": tracks_valid,
            "full_input_checksum_sequences": full_log,
            "expanded_audit_samples": samples,
            "epoch_and_final_hash_roots_equal": roots_equal,
            "pass": tracks_valid and full_log["pass"] and samples["pass"] and roots_equal,
        }
        all_valid = all_valid and fairness_seeds[str(seed)]["pass"]
    fairness = {
        "schema_version": "pillarhist-r8-fairness-audit-v1",
        "created_utc": utc_now(),
        "seeds": fairness_seeds,
        "pass": all_valid,
    }
    json_dump(args.r8_root / "R8_fairness_audit.json", fairness)
    if not all_valid:
        decision = {
            "execution_status": "PARTIAL",
            "scientific_label": None,
            "paper_protocol_mismatch": True,
            "reason": "one or more validity/fairness gates failed",
        }
        json_dump(args.r8_root / "R8_decision.json", decision)
        raise SystemExit(json.dumps(decision))

    metrics = metric_payload(audits)
    json_dump(args.r8_root / "R8_metrics_by_seed.json", metrics)
    paper = paper_comparison(metrics)
    json_dump(args.r8_root / "R8_paper_comparison.json", paper)
    car = metrics["paired_delta_aggregate"]["Car"]["mean_easy_moderate_hard"]
    ped = metrics["paired_delta_aggregate"]["Pedestrian"]["mean_easy_moderate_hard"]
    label = scientific_label(
        car["mean"], ped["mean"], car["positive_seed_count"], ped["positive_seed_count"]
    )
    efficiency = {
        "schema_version": "pillarhist-r8-efficiency-v1",
        "scope": "current-hardware FP32 engineering record; not deployment evidence",
        "runs": {
            f"seed{seed}:{track}": audits[seed][track]["summary"]["runtime"]
            for seed in SEEDS for track in TRACK_SPECS
        },
    }
    json_dump(args.r8_root / "R8_efficiency.json", efficiency)
    decision = {
        "schema_version": "pillarhist-r8-decision-v1",
        "execution_status": "PASS_COMPLETE",
        "scientific_label": label,
        "paper_protocol_mismatch": True,
        "mean_delta_car_map": car["mean"],
        "mean_delta_pedestrian_map": ped["mean"],
        "positive_car_seeds": car["positive_seed_count"],
        "positive_pedestrian_seeds": ped["positive_seed_count"],
        "numeric_source": "unrounded evaluator JSON",
        "created_utc": utc_now(),
    }
    json_dump(args.r8_root / "R8_decision.json", decision)

    lines = [
        "# PillarHist R8 Closure Report", "", "## Decision", "",
        f"R8 closes as `{decision['execution_status']} + {label} + PAPER_PROTOCOL_MISMATCH=true`.",
        "", "## Three-seed paired result", "",
        "| Category | Mean paired delta | Sample std | Positive seeds |",
        "|---|---:|---:|---:|",
        f"| Car three-difficulty mAP | {car['mean']:.6f} | {car['sample_std']:.6f} | {car['positive_seed_count']}/3 |",
        f"| Pedestrian three-difficulty mAP | {ped['mean']:.6f} | {ped['sample_std']:.6f} | {ped['positive_seed_count']}/3 |",
        "", "All labels were computed from unrounded evaluator JSON. No significance claim is made from three seeds.",
        "", "## Validity and fairness", "",
        "- Six 80-epoch runs completed with finite losses, positive anchors, gradient checks, recovery continuity, and full 3,769-frame validation.",
        "- Within every seed, all optimizer-step input checksums, sampled expanded records, epoch roots, and final replay roots match between PP and PH.",
        "- The primary checkpoint is epoch 80 (`LAST_EPOCH`); no best-validation selection was used.",
        "", "## Scope", "",
        "This is a local OpenPCDet FP32 controlled reproduction. Protocol mismatches or unverifiable paper details are recorded separately; the result is not evidence for INT8, TensorRT, or target-hardware speed.",
        "", "## Evidence", "",
        f"- Machine decision: `{args.r8_root / 'R8_decision.json'}`",
        f"- Fairness audit: `{args.r8_root / 'R8_fairness_audit.json'}`",
        f"- Metrics: `{args.r8_root / 'R8_metrics_by_seed.json'}`",
        f"- Paper comparison: `{args.r8_root / 'R8_paper_comparison.json'}`",
        f"- Efficiency: `{args.r8_root / 'R8_efficiency.json'}`", "",
    ]
    report = "\n".join(lines)
    (args.r8_root / "R8_closure_report.md").write_text(report, encoding="utf-8")
    args.docs_report.parent.mkdir(parents=True, exist_ok=True)
    args.docs_report.write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
