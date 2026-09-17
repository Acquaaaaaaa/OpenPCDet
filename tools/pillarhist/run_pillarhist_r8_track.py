#!/usr/bin/env python3
"""Run one frozen 80-epoch R8 track with exact shared replay and recovery."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import pickle
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.nn.utils import clip_grad_norm_

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from r8_protocol import (
    HASH_CHAIN_INITIAL,
    SCHEMA_VERSION,
    TRACK_SPECS,
    GpuTelemetry,
    apply_canonical_initialization,
    batch_checksums,
    build_dataset,
    configure_determinism,
    configure_track,
    cpu_state_dict,
    db_sampler_state,
    environment_record,
    json_dump,
    json_load,
    materialize_replay_batch,
    read_replay,
    representative_gradients,
    restore_db_sampler_state,
    restore_rng_state,
    rng_state,
    sha256_file,
    tensor_dict_checksum,
    to_plain,
    update_hash_chain,
    utc_now,
)


def optimizer_lr(optimizer):
    try:
        return float(optimizer.lr)
    except Exception:
        return float(optimizer.param_groups[0]["lr"])


def optimizer_momentum(optimizer):
    try:
        return float(optimizer.mom)
    except Exception:
        group = optimizer.param_groups[0]
        return float(group.get("momentum", group.get("betas", (float("nan"),))[0]))


def save_checkpoint(path, model, optimizer, dataset, track, seed, completed_epoch,
                    completed_steps, replay_sha256, canonical_sha256, hash_chain,
                    epoch_roots, total_steps):
    payload = {
        "schema_version": f"{SCHEMA_VERSION}-checkpoint",
        "track": track,
        "seed": seed,
        "completed_epoch": completed_epoch,
        "next_epoch_zero_based": completed_epoch,
        "next_batch_zero_based": 0,
        "completed_optimizer_steps": completed_steps,
        "total_optimizer_steps": total_steps,
        "model_state": cpu_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": {
            "type": "OpenPCDet OneCycle",
            "completed_optimizer_steps": completed_steps,
            "total_optimizer_steps": total_steps,
            "lr": optimizer_lr(optimizer),
            "momentum": optimizer_momentum(optimizer),
        },
        "amp_enabled": False,
        "grad_scaler_state": "disabled",
        "rng_state": rng_state(),
        "db_sampler_state": db_sampler_state(dataset),
        "sampler_state": {"next_epoch_zero_based": completed_epoch, "next_batch_zero_based": 0},
        "input_hash_chain": hash_chain,
        "epoch_hash_roots": copy.deepcopy(epoch_roots),
        "replay_manifest_sha256": replay_sha256,
        "canonical_initialization_sha256": canonical_sha256,
        "saved_utc": utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_checkpoint(path, model, optimizer, dataset, replay_sha256, canonical_sha256):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["replay_manifest_sha256"] != replay_sha256:
        raise AssertionError("checkpoint replay hash mismatch")
    if payload["canonical_initialization_sha256"] != canonical_sha256:
        raise AssertionError("checkpoint canonical initialization hash mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng_state(payload["rng_state"])
    restore_db_sampler_state(dataset, payload["db_sampler_state"])
    return payload


def run_update(model, optimizer, scheduler, dataset, entry, config, collect_gradient,
               compute_model_checksum=False):
    from pcdet.models import load_data_to_gpu

    batch, augmentation_matrices, frame_ids = materialize_replay_batch(dataset, entry)
    checksums = batch_checksums(batch)
    load_data_to_gpu(batch)
    scheduler.step(entry["global_step_zero_based"], entry["epoch_one_based"] - 1)
    lr = optimizer_lr(optimizer)
    momentum = optimizer_momentum(optimizer)
    optimizer.zero_grad()
    model.train()
    result, tb, _ = model(batch)
    loss = result["loss"]
    components = {key: float(value) for key, value in tb.items()}
    positive_anchors = int((model.dense_head.forward_ret_dict["box_cls_labels"] > 0).sum())
    if not torch.isfinite(loss) or not all(math.isfinite(value) for value in components.values()):
        raise FloatingPointError({"entry": entry, "loss": float(loss.detach().cpu()), "components": components})
    if positive_anchors <= 0:
        raise AssertionError({"entry": entry, "positive_anchors": positive_anchors})
    loss.backward()
    gradient = representative_gradients(model) if collect_gradient else None
    if gradient is not None and not gradient["pass"]:
        raise AssertionError({"entry": entry, "gradient": gradient})
    gradient_norm = float(
        clip_grad_norm_(model.parameters(), config.OPTIMIZATION.GRAD_NORM_CLIP).detach().cpu()
    )
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "loss_components": components,
        "positive_anchors": positive_anchors,
        "gradient": gradient,
        "gradient_norm_before_clip": gradient_norm,
        "lr": lr,
        "momentum": momentum,
        "checksums": checksums,
        "augmentation_matrices": augmentation_matrices,
        "frame_ids": frame_ids,
        "model_checksum_after_update": (
            tensor_dict_checksum(model.state_dict()) if compute_model_checksum else None
        ),
    }


def recovery_continuity_check(checkpoint_path, model, optimizer, scheduler, dataset,
                              entry, config, replay_sha256, canonical_sha256):
    uninterrupted = run_update(
        model, optimizer, scheduler, dataset, entry, config,
        collect_gradient=True, compute_model_checksum=True,
    )
    load_checkpoint(
        checkpoint_path, model, optimizer, dataset, replay_sha256, canonical_sha256
    )
    resumed = run_update(
        model, optimizer, scheduler, dataset, entry, config,
        collect_gradient=True, compute_model_checksum=True,
    )
    component_keys = sorted(set(uninterrupted["loss_components"]) | set(resumed["loss_components"]))
    component_match = all(
        key in uninterrupted["loss_components"]
        and key in resumed["loss_components"]
        and math.isclose(
            uninterrupted["loss_components"][key], resumed["loss_components"][key],
            rel_tol=1e-5, abs_tol=1e-6,
        )
        for key in component_keys
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "next_epoch_one_based": entry["epoch_one_based"],
        "next_batch_zero_based": entry["batch_zero_based"],
        "next_global_step_zero_based": entry["global_step_zero_based"],
        "input_checksum_equal": uninterrupted["checksums"]["input"] == resumed["checksums"]["input"],
        "lr_equal": uninterrupted["lr"] == resumed["lr"],
        "momentum_equal": uninterrupted["momentum"] == resumed["momentum"],
        "loss_close": math.isclose(uninterrupted["loss"], resumed["loss"], rel_tol=1e-5, abs_tol=1e-6),
        "loss_components_close": component_match,
        "positive_anchors_equal": uninterrupted["positive_anchors"] == resumed["positive_anchors"],
        "gradient_audit_equal": uninterrupted["gradient"] == resumed["gradient"],
        "model_checksum_after_update_equal": (
            uninterrupted["model_checksum_after_update"] == resumed["model_checksum_after_update"]
        ),
        "rtol": 1e-5,
        "atol": 1e-6,
        "byte_exact_model_update_required": True,
    }
    result["pass"] = all(
        value for key, value in result.items()
        if key.endswith("_equal") or key.endswith("_close")
    )
    load_checkpoint(
        checkpoint_path, model, optimizer, dataset, replay_sha256, canonical_sha256
    )
    if not result["pass"]:
        raise AssertionError(result)
    return result


@torch.no_grad()
def evaluate(model, dataset, config, output_dir, batch_size):
    from pcdet.models import load_data_to_gpu

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    class_names = list(config.CLASS_NAMES)
    det_annos = []
    prediction_count = 0
    recall = {f"roi_{value}": 0 for value in config.MODEL.POST_PROCESSING.RECALL_THRESH_LIST}
    recall.update({f"rcnn_{value}": 0 for value in config.MODEL.POST_PROCESSING.RECALL_THRESH_LIST})
    recall["gt"] = 0
    started = time.time()
    for offset in range(0, len(dataset), batch_size):
        indices = list(range(offset, min(offset + batch_size, len(dataset))))
        batch = dataset.collate_batch([dataset[index] for index in indices])
        load_data_to_gpu(batch)
        predictions, returned = model(batch)
        for key in recall:
            recall[key] += int(returned.get(key, 0))
        prediction_count += sum(int(item["pred_boxes"].shape[0]) for item in predictions)
        det_annos.extend(dataset.generate_prediction_dicts(batch, predictions, class_names))
        if offset % 200 == 0:
            print(json.dumps({
                "phase": "full_validation",
                "frames_done": min(offset + batch_size, len(dataset)),
                "frames_total": len(dataset),
                "utc": utc_now(),
            }), flush=True)
    result_text, metrics = dataset.evaluation(
        det_annos, class_names, eval_metric=config.MODEL.POST_PROCESSING.EVAL_METRIC
    )
    metrics = {key: float(value) for key, value in metrics.items()}
    class_summary = {}
    for name in class_names:
        values = {
            difficulty: metrics[f"{name}_3d/{difficulty}_R40"]
            for difficulty in ("easy", "moderate", "hard")
        }
        values["mean_easy_moderate_hard"] = statistics.fmean(values.values())
        class_summary[name] = values
    recall_rates = {}
    for threshold in config.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        recall_rates[f"roi_{threshold}"] = recall[f"roi_{threshold}"] / max(recall["gt"], 1)
        recall_rates[f"rcnn_{threshold}"] = recall[f"rcnn_{threshold}"] / max(recall["gt"], 1)
    elapsed = time.time() - started
    summary = {
        "frames": len(dataset),
        "prediction_count": prediction_count,
        "recall_counts": recall,
        "recall_rates": recall_rates,
        "classes": class_summary,
        "three_class_moderate_macro": statistics.fmean(
            class_summary[name]["moderate"] for name in class_names
        ),
        "elapsed_seconds": elapsed,
        "seconds_per_frame": elapsed / len(dataset),
        "metrics_raw_unrounded": metrics,
    }
    json_dump(output_dir / "metrics.json", summary)
    (output_dir / "result.txt").write_text(result_text, encoding="utf-8")
    with (output_dir / "result.pkl").open("wb") as stream:
        pickle.dump(det_annos, stream)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--track", choices=TRACK_SPECS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--telemetry-interval", type=float, default=30.0)
    parser.add_argument("--skip-validation", action="store_true", help="Exploratory use only; forbidden for closure")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R8 requires CUDA")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.resume is None and any(args.output_root.iterdir()):
        raise FileExistsError(f"non-empty output root without --resume: {args.output_root}")

    repo = Path(__file__).resolve().parents[2]
    prep = json_load(args.preparation_root / "preparation_manifest.json")
    seed = int(prep["seed"])
    batch_size = int(prep["batch_size"])
    epochs = int(prep["epochs"])
    determinism = configure_determinism(seed)
    config = configure_track(repo, args.track, batch_size, epochs)
    train_dataset, train_loader = build_dataset(config, True, seed, batch_size)
    validation_dataset, _ = build_dataset(config, False, seed, batch_size)
    replay_manifest_path = args.preparation_root / "replay_manifest.json"
    replay_manifest = json_load(replay_manifest_path)
    replay_path = args.preparation_root / "replay_steps.jsonl.gz"
    if sha256_file(replay_path) != replay_manifest["replay_steps"]["sha256"]:
        raise AssertionError("replay steps hash mismatch")
    replay_entries = list(read_replay(replay_path))
    if len(replay_entries) != prep["total_optimizer_steps"]:
        raise AssertionError("replay entry count mismatch")

    from pcdet.models import build_network
    from train_utils.optimization import build_optimizer, build_scheduler

    torch.manual_seed(seed)
    model = build_network(config.MODEL, len(config.CLASS_NAMES), train_dataset)
    canonical_path = args.preparation_root / "canonical_initialization.pth"
    canonical_sha256 = sha256_file(canonical_path)
    package = torch.load(canonical_path, map_location="cpu", weights_only=False)
    initialization = apply_canonical_initialization(model, package, args.track)
    optimizer = build_optimizer(model, config.OPTIMIZATION)
    scheduler, warmup = build_scheduler(
        optimizer,
        total_iters_each_epoch=replay_manifest["steps_per_epoch"],
        total_epochs=epochs,
        last_epoch=-1,
        optim_cfg=config.OPTIMIZATION,
    )
    if warmup is not None:
        raise AssertionError("R8 frozen protocol forbids LR warmup")
    model.cuda().train()
    replay_sha256 = sha256_file(replay_manifest_path)

    start_epoch = 0
    completed_steps = 0
    hash_chain = HASH_CHAIN_INITIAL
    epoch_roots = {}
    resume_from = None
    recovery_result = None
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume, model, optimizer, train_dataset, replay_sha256, canonical_sha256
        )
        start_epoch = int(checkpoint["next_epoch_zero_based"])
        completed_steps = int(checkpoint["completed_optimizer_steps"])
        hash_chain = checkpoint["input_hash_chain"]
        epoch_roots = checkpoint["epoch_hash_roots"]
        resume_from = str(args.resume)
        recovery_path = args.output_root / "diagnostics/resume_continuity.json"
        if recovery_path.exists():
            recovery_result = json_load(recovery_path)

    resolved_path = args.output_root / "config_resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(to_plain(config), sort_keys=False), encoding="utf-8")
    env = environment_record(repo)
    (args.output_root / "environment.txt").write_text(json.dumps(env, indent=2), encoding="utf-8")
    command_dir = args.output_root / "commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    (command_dir / "argv.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    run_manifest = {
        "schema_version": f"{SCHEMA_VERSION}-track-run",
        "status": "RUNNING",
        "track": args.track,
        "seed": seed,
        "started_utc": utc_now(),
        "batch_size": batch_size,
        "global_batch_size": batch_size,
        "workers": 0,
        "gradient_accumulation": 1,
        "epochs": epochs,
        "steps_per_epoch": replay_manifest["steps_per_epoch"],
        "total_optimizer_steps": replay_manifest["total_optimizer_steps"],
        "amp": False,
        "determinism": determinism,
        "initialization": initialization,
        "optimizer_created_after_canonical_copy": True,
        "replay_manifest_sha256": replay_sha256,
        "canonical_initialization_sha256": canonical_sha256,
        "git_head": env["git_head"],
        "git_branch": env["git_branch"],
        "runner_sha256": sha256_file(__file__),
        "config_resolved_sha256": sha256_file(resolved_path),
        "resume_from": resume_from,
    }
    json_dump(args.output_root / "manifest.json", run_manifest)

    telemetry = GpuTelemetry(args.output_root / "gpu_telemetry.csv", args.telemetry_interval)
    telemetry.start()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    elapsed_offset = 0.0
    if start_epoch and (args.output_root / "training_log.jsonl").exists():
        prior_records = [
            json.loads(line)
            for line in (args.output_root / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        if prior_records:
            elapsed_offset = float(prior_records[-1]["elapsed_seconds"])
    log_path = args.output_root / "training_log.jsonl"
    sample_path = args.output_root / "replay_samples.jsonl.gz"
    log_mode = "a" if start_epoch else "w"
    sample_mode = "at" if start_epoch else "wt"
    gradient_checks = []
    checkpoint_records = []
    existing_checkpoint_manifest = args.output_root / "checkpoint_manifest.json"
    if start_epoch and existing_checkpoint_manifest.exists():
        checkpoint_records = json_load(existing_checkpoint_manifest).get("checkpoints", [])
    try:
        with log_path.open(log_mode, encoding="utf-8") as log_stream, gzip.open(
            sample_path, sample_mode, encoding="utf-8", newline="\n"
        ) as sample_stream:
            for epoch_zero_based in range(start_epoch, epochs):
                epoch_one_based = epoch_zero_based + 1
                epoch_start = epoch_zero_based * replay_manifest["steps_per_epoch"]
                epoch_end = epoch_start + replay_manifest["steps_per_epoch"]
                epoch_entries = replay_entries[epoch_start:epoch_end]
                audit_batches = set(
                    replay_manifest["audit_batch_indices_zero_based"][str(epoch_one_based)]
                )
                for entry in epoch_entries:
                    collect_gradient = entry["optimizer_step_one_based"] % 100 == 0
                    result = run_update(
                        model, optimizer, scheduler, train_dataset, entry, config, collect_gradient
                    )
                    completed_steps = entry["optimizer_step_one_based"]
                    hash_chain = update_hash_chain(
                        hash_chain, entry["global_step_zero_based"], result["checksums"]["input"]
                    )
                    if collect_gradient:
                        gradient_checks.append({
                            "optimizer_step_one_based": completed_steps,
                            **result["gradient"],
                        })
                    record = {
                        "epoch_one_based": epoch_one_based,
                        "batch_zero_based": entry["batch_zero_based"],
                        "optimizer_step_one_based": completed_steps,
                        "loss": result["loss"],
                        "loss_components": result["loss_components"],
                        "lr": result["lr"],
                        "momentum": result["momentum"],
                        "positive_anchors": result["positive_anchors"],
                        "gradient_norm_before_clip": result["gradient_norm_before_clip"],
                        "gradient": result["gradient"],
                        "input_checksum": result["checksums"]["input"],
                        "input_hash_chain": hash_chain,
                        "elapsed_seconds": elapsed_offset + time.time() - started,
                    }
                    log_stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    if entry["batch_zero_based"] in audit_batches:
                        sample_stream.write(json.dumps({
                            "epoch_one_based": epoch_one_based,
                            "batch_zero_based": entry["batch_zero_based"],
                            "optimizer_step_one_based": completed_steps,
                            "dataset_indices": entry["dataset_indices"],
                            "sample_seeds": entry["sample_seeds"],
                            "frame_ids": result["frame_ids"],
                            "augmentation_matrices": result["augmentation_matrices"],
                            "checksums": result["checksums"],
                        }, separators=(",", ":")) + "\n")
                    if completed_steps % 100 == 0 or entry["batch_zero_based"] == 0:
                        log_stream.flush()
                        sample_stream.flush()
                        print(json.dumps({
                            "phase": "train",
                            "track": args.track,
                            "seed": seed,
                            "epoch": epoch_one_based,
                            "batch": entry["batch_zero_based"],
                            "step": completed_steps,
                            "loss": result["loss"],
                            "lr": result["lr"],
                            "elapsed_seconds": elapsed_offset + time.time() - started,
                            "utc": utc_now(),
                        }), flush=True)
                epoch_roots[str(epoch_one_based)] = hash_chain
                if epoch_one_based % 10 == 0:
                    checkpoint_path = args.output_root / "checkpoints" / f"epoch_{epoch_one_based:03d}.pth"
                    save_checkpoint(
                        checkpoint_path, model, optimizer, train_dataset, args.track, seed,
                        epoch_one_based, completed_steps, replay_sha256, canonical_sha256,
                        hash_chain, epoch_roots, replay_manifest["total_optimizer_steps"],
                    )
                    checkpoint_records.append({
                        "epoch": epoch_one_based,
                        "path": str(checkpoint_path),
                        "sha256": sha256_file(checkpoint_path),
                    })
                    json_dump(args.output_root / "checkpoint_manifest.json", {
                        "schema_version": f"{SCHEMA_VERSION}-checkpoint-manifest",
                        "checkpoints": checkpoint_records,
                    })
                    if epoch_one_based == 10 and recovery_result is None:
                        next_entry = replay_entries[completed_steps]
                        recovery_result = recovery_continuity_check(
                            checkpoint_path, model, optimizer, scheduler, train_dataset,
                            next_entry, config, replay_sha256, canonical_sha256,
                        )
                        json_dump(
                            args.output_root / "diagnostics/resume_continuity.json",
                            recovery_result,
                        )
        final_checkpoint = args.output_root / "checkpoints" / "epoch_080.pth"
        shutil.copy2(final_checkpoint, args.output_root / "checkpoints" / "last.pth")
        checkpoint_records.append({
            "epoch": 80,
            "role": "LAST_EPOCH",
            "path": str(args.output_root / "checkpoints" / "last.pth"),
            "sha256": sha256_file(args.output_root / "checkpoints" / "last.pth"),
        })
        json_dump(args.output_root / "checkpoint_manifest.json", {
            "schema_version": f"{SCHEMA_VERSION}-checkpoint-manifest",
            "checkpoints": checkpoint_records,
        })
        validation = None
        if not args.skip_validation:
            validation = evaluate(
                model, validation_dataset, config,
                args.output_root / "validation/full", batch_size,
            )
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
        gradient_checks = [
            {
                "optimizer_step_one_based": record["optimizer_step_one_based"],
                **record["gradient"],
            }
            for record in records if record.get("gradient") is not None
        ]
        total_training_wall = records[-1]["elapsed_seconds"] if records else 0.0
        runtime = {
            "training_wall_seconds": total_training_wall,
            "optimizer_steps": completed_steps,
            "optimizer_steps_per_second": completed_steps / max(total_training_wall, 1e-9),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "checkpoint_bytes": (args.output_root / "checkpoints" / "last.pth").stat().st_size,
        }
        json_dump(args.output_root / "runtime_and_memory.json", runtime)
        json_dump(args.output_root / "gradient_audit.json", {
            "parameter_policy": "projection(if present), first trainable 2D-backbone weight, dense_head.conv_cls.weight",
            "sampling_interval_optimizer_steps": 100,
            "requirements": {"present": True, "finite": True, "nonzero": True},
            "consecutive_failures_before_failure": 1,
            "checks": gradient_checks,
            "pass": bool(gradient_checks) and all(item["pass"] for item in gradient_checks),
        })
        json_dump(args.output_root / "replay_hash_chain.json", {
            "schema_version": f"{SCHEMA_VERSION}-input-hash-chain",
            "initial_root": HASH_CHAIN_INITIAL,
            "epoch_roots": epoch_roots,
            "final_root": hash_chain,
            "steps": completed_steps,
        })
        summary = {
            "status": "COMPLETE" if not args.skip_validation else "EXPLORATORY_COMPLETE",
            "track": args.track,
            "seed": seed,
            "completed_epochs": epochs,
            "completed_optimizer_steps": completed_steps,
            "all_losses_finite": all(math.isfinite(record["loss"]) for record in records),
            "all_positive_anchor_counts_above_zero": all(record["positive_anchors"] > 0 for record in records),
            "gradient_checks_pass": bool(gradient_checks) and all(item["pass"] for item in gradient_checks),
            "resume_continuity": recovery_result,
            "input_hash_chain_final_root": hash_chain,
            "loss_first_200_median": statistics.median(record["loss"] for record in records[:200]),
            "loss_last_200_median": statistics.median(record["loss"] for record in records[-200:]),
            "validation": validation,
            "runtime": runtime,
            "final_model_checksum": tensor_dict_checksum(model.state_dict()),
            "finished_utc": utc_now(),
        }
        json_dump(args.output_root / "track_summary.json", summary)
        run_manifest["status"] = summary["status"]
        run_manifest["finished_utc"] = summary["finished_utc"]
        json_dump(args.output_root / "manifest.json", run_manifest)
        print(json.dumps({
            "status": summary["status"],
            "track": args.track,
            "seed": seed,
            "final_root": hash_chain,
            "validation": validation,
        }), flush=True)
    finally:
        telemetry.stop()


if __name__ == "__main__":
    main()
