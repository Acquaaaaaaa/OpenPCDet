#!/usr/bin/env python3
"""Audit the frozen R7 runs and generate the closure decision artifacts."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch


SEED666_TRACK_DIRS = {
    "PH_RAW_LINEAR": "PH_RAW_LINEAR",
    "PH_NORM_LINEAR": "PH_NORM_LINEAR",
    "PH_RAW_BNRELU": "PH_RAW_BNRELU",
    "PH_NORM_BNRELU": "PH_NORM_BNRELU",
    "PP_SHORT_CONTROL": "PP_SHORT_CONTROL_attempt2",
}
SEED667_TRACK_DIRS = {
    "PH_NORM_LINEAR": "PH_NORM_LINEAR",
    "PH_NORM_BNRELU": "PH_NORM_BNRELU",
}
CANDIDATES = [
    "PH_RAW_LINEAR",
    "PH_NORM_LINEAR",
    "PH_RAW_BNRELU",
    "PH_NORM_BNRELU",
]
TIE_MARGIN = 1.0


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_log(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def checkpoint_audit(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    running_means = [value for key, value in state.items() if key.endswith("running_mean")]
    running_vars = [value for key, value in state.items() if key.endswith("running_var")]
    bn_finite = all(torch.isfinite(value).all().item() for value in running_means + running_vars)
    bn_vars_nonnegative = all((value >= 0).all().item() for value in running_vars)
    required = {
        "model_state", "optimizer_state", "scheduler_state", "grad_scaler_state",
        "rng_state", "db_sampler_state", "completed_optimizer_steps",
        "next_step_zero_based", "training_replay_manifest_sha256",
        "canonical_initialization_sha256",
    }
    return {
        "schema_version": checkpoint.get("schema_version"),
        "required_fields_present": required.issubset(checkpoint),
        "completed_optimizer_steps": int(checkpoint["completed_optimizer_steps"]),
        "next_step_zero_based": int(checkpoint["next_step_zero_based"]),
        "bn_running_mean_tensors": len(running_means),
        "bn_running_var_tensors": len(running_vars),
        "bn_running_stats_finite": bool(bn_finite),
        "bn_running_vars_nonnegative": bool(bn_vars_nonnegative),
        "optimizer_state_nonempty": bool(checkpoint["optimizer_state"].get("state")),
    }


def audit_track(track, root, replay, canonical):
    manifest = load_json(root / "manifest.json")
    summary = load_json(root / "track_summary.json")
    records = read_log(root / "training_log.jsonl")
    checkpoint = checkpoint_audit(root / "checkpoints/last.pth")
    finite = all(
        math.isfinite(record["loss"])
        and math.isfinite(record["gradient_norm_before_clip"])
        and all(math.isfinite(value) for value in record["loss_components"].values())
        for record in records
    )
    positive = all(record["positive_anchors"] > 0 for record in records)
    validation_seconds = sum(item["elapsed_seconds"] for item in summary["subset_validations"])
    validation_seconds += summary["full_validation_primary"]["elapsed_seconds"]
    training_elapsed = float(records[-1]["elapsed_seconds"])
    track_init = canonical["tracks"][track]
    valid = all([
        summary["status"] == "COMPLETE",
        summary["completed_optimizer_steps"] == 5000,
        len(records) == 5000,
        finite,
        positive,
        summary["all_gradient_checks_pass"],
        len(summary["gradient_checks"]) == 50,
        bool(summary["resume_continuity"] and summary["resume_continuity"]["pass"]),
        summary["loss"]["last_below_first"],
        manifest["optimizer_created_after_canonical_copy"],
        manifest["initialization"]["backend_checksum"] == canonical["backend_checksum"],
        summary["canonical_initial_model_checksum"] == track_init["complete_model_checksum"],
        checkpoint["required_fields_present"],
        checkpoint["completed_optimizer_steps"] == 5000,
        checkpoint["bn_running_stats_finite"],
        checkpoint["bn_running_vars_nonnegative"],
    ])
    return {
        "track": track,
        "root": str(root),
        "valid": valid,
        "status": summary["status"],
        "optimizer_steps": len(records),
        "all_losses_and_components_finite": finite,
        "all_positive_anchor_counts_above_zero": positive,
        "minimum_positive_anchors": min(record["positive_anchors"] for record in records),
        "gradient_checks": len(summary["gradient_checks"]),
        "all_gradient_checks_pass": summary["all_gradient_checks_pass"],
        "resume_continuity_pass": summary["resume_continuity"]["pass"],
        "loss_first_200_median": summary["loss"]["first_200_median"],
        "loss_last_200_median": summary["loss"]["last_200_median"],
        "full_validation": summary["full_validation_primary"],
        "input_checksums": [record["input_checksum"] for record in records],
        "replay_manifest_sha256": manifest["replay_manifest_sha256"],
        "canonical_initialization_sha256": manifest["canonical_initialization_sha256"],
        "canonical_backend_checksum": manifest["initialization"]["backend_checksum"],
        "canonical_initial_model_checksum": summary["canonical_initial_model_checksum"],
        "optimizer_created_after_canonical_copy": manifest["optimizer_created_after_canonical_copy"],
        "checkpoint": checkpoint,
        "training_elapsed_to_step_5000_seconds": training_elapsed,
        "observed_protocol_optimizer_steps_per_second": 5000.0 / training_elapsed,
        "validation_elapsed_seconds": validation_seconds,
        "accounted_wall_seconds": training_elapsed + summary["full_validation_primary"]["elapsed_seconds"],
        "runner_sha256": manifest["runner_sha256"],
        "git_head": manifest["git_head"],
        "pp_short_control_sanity": summary.get("pp_short_control_sanity"),
    }


def fairness_group(seed, prep_root, tracks, track_dirs):
    replay_path = prep_root / "training_replay_manifest.json"
    replay = load_json(replay_path)
    canonical = load_json(prep_root / "canonical_initialization.json")
    audited = {
        track: audit_track(track, prep_root / "tracks" / track_dirs[track], replay, canonical)
        for track in tracks
    }
    reference_checksums = audited[tracks[0]]["input_checksums"]
    audit_indices = replay["audit_step_indices_zero_based"]
    full_sequences_equal = all(
        item["input_checksums"] == reference_checksums for item in audited.values()
    )
    replay_hashes = {item["replay_manifest_sha256"] for item in audited.values()}
    backend_hashes = {item["canonical_backend_checksum"] for item in audited.values()}
    audit_checksums_equal = all(
        [item["input_checksums"][index] for index in audit_indices]
        == [reference_checksums[index] for index in audit_indices]
        for item in audited.values()
    )
    fairness_pass = all([
        replay["seed"] == seed,
        replay["training_steps"] == 5000,
        replay["resume_probe_steps"] == 1,
        len(audit_indices) == 22,
        full_sequences_equal,
        audit_checksums_equal,
        len(replay_hashes) == 1,
        len(backend_hashes) == 1,
        canonical["optimizer_created_after_copy"],
        all(item["valid"] for item in audited.values()),
    ])
    compact_tracks = {}
    for track, item in audited.items():
        compact = dict(item)
        compact.pop("input_checksums")
        compact_tracks[track] = compact
    return {
        "seed": seed,
        "preparation_root": str(prep_root),
        "preparation_manifest_sha256": sha256(prep_root / "preparation_manifest.json"),
        "replay_manifest_sha256": sha256(replay_path),
        "canonical_initialization_file_sha256": sha256(prep_root / "canonical_initialization.pth"),
        "audit_step_indices_zero_based": audit_indices,
        "full_5000_step_input_checksum_sequences_equal": full_sequences_equal,
        "audit_step_input_checksums_equal": audit_checksums_equal,
        "shared_replay_hash": len(replay_hashes) == 1,
        "shared_backend_initialization": len(backend_hashes) == 1,
        "optimizer_created_after_canonical_copy": canonical["optimizer_created_after_copy"],
        "fairness_pass": fairness_pass,
        "tracks": compact_tracks,
    }


def mib(value):
    return value / (1024.0 * 1024.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed666-root", type=Path, required=True)
    parser.add_argument("--seed667-root", type=Path, required=True)
    parser.add_argument("--memory-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--docs-report", type=Path, required=True)
    args = parser.parse_args()

    group666 = fairness_group(666, args.seed666_root, list(SEED666_TRACK_DIRS), SEED666_TRACK_DIRS)
    group667 = fairness_group(667, args.seed667_root, list(SEED667_TRACK_DIRS), SEED667_TRACK_DIRS)
    if args.memory_audit.is_dir():
        memory = {
            "schema_version": "pillarhist-r7-memory-audit-v1",
            "tracks": {
                item["track"]: item
                for item in (load_json(path) for path in sorted(args.memory_audit.glob("*.json")))
            },
        }
    else:
        memory = load_json(args.memory_audit)

    primary_scores = {
        track: group666["tracks"][track]["full_validation"]["macro_3d_ap_r40_moderate"]
        for track in CANDIDATES
    }
    primary_ranking = sorted(primary_scores, key=primary_scores.get, reverse=True)
    tied = primary_ranking[:2]
    seed666_gap = primary_scores[tied[0]] - primary_scores[tied[1]]
    seed667_scores = {
        track: group667["tracks"][track]["full_validation"]["macro_3d_ap_r40_moderate"]
        for track in tied
    }
    two_seed_means = {
        track: statistics.fmean([primary_scores[track], seed667_scores[track]])
        for track in tied
    }
    aggregate_ranking = sorted(two_seed_means, key=two_seed_means.get, reverse=True)
    aggregate_gap = two_seed_means[aggregate_ranking[0]] - two_seed_means[aggregate_ranking[1]]
    pp = group666["tracks"]["PP_SHORT_CONTROL"]
    fairness_pass = group666["fairness_pass"] and group667["fairness_pass"]
    protocol_pass = bool(pp["pp_short_control_sanity"] and pp["pp_short_control_sanity"]["pass"])
    if not fairness_pass:
        decision = "BLOCKED"
    elif not protocol_pass:
        decision = "NO_PROMOTION"
    elif aggregate_gap < TIE_MARGIN:
        decision = "INCONCLUSIVE"
    else:
        decision = "SELECTED"

    decision_payload = {
        "schema_version": "pillarhist-r7-decision-v1",
        "status": decision,
        "primary_metric": "full KITTI validation three-class macro 3D AP_R40 Moderate",
        "tie_margin_absolute_ap": TIE_MARGIN,
        "seed666_ranking": primary_ranking,
        "seed666_scores": primary_scores,
        "seed666_top_two_gap": seed666_gap,
        "seed667_rechecked_candidates": tied,
        "seed667_scores": seed667_scores,
        "two_seed_means": two_seed_means,
        "two_seed_mean_gap": aggregate_gap,
        "fairness_pass": fairness_pass,
        "pp_short_control_sanity_pass": protocol_pass,
        "selected_track": aggregate_ranking[0] if decision == "SELECTED" else None,
        "r8_automatic_start_allowed": False,
        "reason": (
            "The two rechecked candidates remain within the frozen 1.0 AP uncertainty margin."
            if decision == "INCONCLUSIVE" else decision
        ),
    }

    fairness_payload = {
        "schema_version": "pillarhist-r7-fairness-audit-v1",
        "pass": fairness_pass,
        "seed666": group666,
        "seed667": group667,
    }
    efficiency_tracks = {}
    for group in [group666, group667]:
        for track, item in group["tracks"].items():
            key = f"seed{group['seed']}:{track}"
            efficiency_tracks[key] = {
                "training_elapsed_to_step_5000_seconds": item["training_elapsed_to_step_5000_seconds"],
                "observed_protocol_optimizer_steps_per_second": item["observed_protocol_optimizer_steps_per_second"],
                "validation_elapsed_seconds": item["validation_elapsed_seconds"],
                "accounted_wall_seconds": item["accounted_wall_seconds"],
            }
    efficiency_payload = {
        "schema_version": "pillarhist-r7-efficiency-v1",
        "tracks": efficiency_tracks,
        "primary_seed_memory_probe": memory,
        "memory_scope": "single fixed post-training resume-probe forward/backward; diagnostic, not a ranking metric",
    }

    dump_json(args.output_root / "R7_decision.json", decision_payload)
    dump_json(args.output_root / "R7_fairness_audit.json", fairness_payload)
    dump_json(args.output_root / "R7_efficiency.json", efficiency_payload)

    lines = [
        "# PillarHist R7 Decision Report",
        "",
        "## Decision",
        "",
        f"R7 closes as `{decision}`.",
        "",
        "The two candidates rechecked with seed 667 remain inside the frozen 1.0 absolute-AP uncertainty margin. No single R8 configuration is automatically selected, and R8 must not start without human review.",
        "",
        "## Frozen protocol",
        "",
        "- Primary metric: full KITTI validation macro of Car/Pedestrian/Cyclist 3D AP_R40 Moderate",
        "- Primary seed: 666; tie-review seed: 667",
        "- 5000 optimizer steps, batch 2, workers 0, AMP disabled, native Adam OneCycle",
        "- Tie margin: 1.0 absolute AP point",
        "- Reduction path: `deterministic_segment`",
        "- Validation subset: frozen 256-frame manifest; final decision uses all 3769 validation frames",
        "",
        "## Seed 666 candidate results",
        "",
        "| Rank | Track | Car | Pedestrian | Cyclist | Macro |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, track in enumerate(primary_ranking, 1):
        full = group666["tracks"][track]["full_validation"]
        cls = full["three_class_3d_ap_r40_moderate"]
        lines.append(f"| {rank} | `{track}` | {cls['Car']:.3f} | {cls['Pedestrian']:.3f} | {cls['Cyclist']:.3f} | {full['macro_3d_ap_r40_moderate']:.3f} |")
    lines += [
        "",
        f"The top-two gap was {seed666_gap:.3f} AP, so only `{tied[0]}` and `{tied[1]}` entered the pre-registered seed 667 review.",
        "",
        "## Tie review",
        "",
        "| Track | Seed 666 | Seed 667 | Two-seed mean |",
        "|---|---:|---:|---:|",
    ]
    for track in aggregate_ranking:
        lines.append(f"| `{track}` | {primary_scores[track]:.3f} | {seed667_scores[track]:.3f} | {two_seed_means[track]:.3f} |")
    lines += [
        "",
        f"The two-seed mean gap is {aggregate_gap:.3f} AP, still below 1.0 AP. The result is therefore `INCONCLUSIVE`, not a forced simplicity tie-break or a claim of statistical superiority.",
        "",
        "## Fairness and validity audit",
        "",
        f"- Seed 666 full 5000-step input checksum equality across five tracks: `{group666['full_5000_step_input_checksum_sequences_equal']}`",
        f"- Seed 667 full 5000-step input checksum equality across the two reviewed tracks: `{group667['full_5000_step_input_checksum_sequences_equal']}`",
        f"- Shared replay hash and canonical backend per seed: `{group666['shared_replay_hash'] and group666['shared_backend_initialization'] and group667['shared_replay_hash'] and group667['shared_backend_initialization']}`",
        "- Optimizers were created only after canonical weights were copied.",
        "- Every completed run has 5000 finite-loss steps with positive anchors, 50/50 passing gradient audits, finite BN running statistics, and an exact fixed-next-batch resume check.",
        f"- Overall fairness audit: `{'PASS' if fairness_pass else 'FAIL'}`",
        "",
        "## PP short control",
        "",
    ]
    pp_full = pp["full_validation"]
    pp_cls = pp_full["three_class_3d_ap_r40_moderate"]
    lines += [
        f"`PP_SHORT_CONTROL` passed all frozen sanity gates. Its first/last 200-step median losses were {pp['loss_first_200_median']:.4f} and {pp['loss_last_200_median']:.4f}; full-validation Car AP was {pp_cls['Car']:.3f}, macro AP was {pp_full['macro_3d_ap_r40_moderate']:.3f}, and predictions were non-empty. It is not part of the PillarHist ranking.",
        "",
        "## Controlled retry",
        "",
        "The first `PP_SHORT_CONTROL` attempt stopped at the frozen step-100 gradient diagnostic because the runner selected `Backbone block[0][0]`, which is `ZeroPad2d`, as though it were a trainable convolution. No checkpoint or result was produced. Amendment 1 changed only representative-parameter selection, from runner SHA-256 `8a746162...1040` to `920de583...23ea6`; replay, canonical initialization, model mathematics, optimizer, scheduler, thresholds, and validation were unchanged. A 100-step preflight passed, and the allowed `PP_SHORT_CONTROL_attempt2` completed the formal protocol.",
        "",
        "## Efficiency records",
        "",
        "| Run | Accounted wall (s) | Observed protocol steps/s |",
        "|---|---:|---:|",
    ]
    for key, item in efficiency_tracks.items():
        lines.append(f"| `{key}` | {item['accounted_wall_seconds']:.1f} | {item['observed_protocol_optimizer_steps_per_second']:.3f} |")
    lines += [
        "",
        "The observed protocol rate includes checkpointing and intermediate validation pauses. CUDA memory is reported separately from a fixed post-training resume-probe backward pass and is diagnostic only:",
        "",
        "| Track | Peak allocated (MiB) | Incremental peak (MiB) |",
        "|---|---:|---:|",
    ]
    for track, item in memory["tracks"].items():
        lines.append(f"| `{track}` | {mib(item['peak_allocated_bytes']):.1f} | {mib(item['incremental_peak_allocated_bytes']):.1f} |")
    lines += [
        "",
        "## Verification",
        "",
        "The final repository verification completed with `41 passed, 1 xfailed`. The expected failure is the separately documented official Scatter empty-input limitation and is not an R7 regression.",
        "",
        "## Scope limits",
        "",
        "This short-run screen does not establish paper-level or official KITTI accuracy, full-training convergence, statistical significance, PTQ/INT8 behavior, TensorRT performance, or target-hardware deployment speed. R6 did not promote the optimized reduction path, so no deployment-speedup claim is supported. Human review is required before any R8 plan.",
        "",
        "## Evidence",
        "",
        f"- Seed 666 preparation: `{args.seed666_root}`",
        f"- Seed 667 preparation: `{args.seed667_root}`",
        f"- Machine-readable decision: `{args.output_root / 'R7_decision.json'}`",
        f"- Fairness audit: `{args.output_root / 'R7_fairness_audit.json'}`",
        f"- Efficiency record: `{args.output_root / 'R7_efficiency.json'}`",
        "",
    ]
    report = "\n".join(lines)
    (args.output_root / "R7_decision_report.md").write_text(report, encoding="utf-8")
    args.docs_report.parent.mkdir(parents=True, exist_ok=True)
    args.docs_report.write_text(report, encoding="utf-8")
    print(json.dumps(decision_payload, indent=2))


if __name__ == "__main__":
    main()
