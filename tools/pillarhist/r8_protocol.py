#!/usr/bin/env python3
"""Shared, frozen protocol helpers for PillarHist R8 full training."""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from prepare_pillarhist_r7 import (
    apply_canonical_initialization as apply_r7_canonical_initialization,
    batch_checksums,
    build_dataset as build_r7_dataset,
    configure_track as configure_r7_track,
    load_config,
    materialize_replay_batch,
    set_seed,
    sha256_file,
    tensor_dict_checksum,
    to_plain,
    utc_now,
)


TRACK_SPECS = {
    "PP_OFFICIAL_FULL": {"r7_track": "PP_SHORT_CONTROL", "family": "pointpillar"},
    "PH_PAPER_LITERAL_FULL": {"r7_track": "PH_RAW_LINEAR", "family": "pillarhist"},
}
SCHEMA_VERSION = "pillarhist-r8-v1"
HASH_CHAIN_INITIAL = hashlib.sha256(b"pillarhist-r8-input-chain-v1").hexdigest()
PAPER_VALUES = {
    "PointPillars": {
        "Car": {"easy": 87.08, "moderate": 77.90, "hard": 74.97, "paper_reported_map": 79.98},
        "Pedestrian": {"easy": 54.71, "moderate": 49.01, "hard": 44.52, "paper_reported_map": 49.41},
    },
    "PH-PointPillars": {
        "Car": {"easy": 88.80, "moderate": 79.13, "hard": 76.30, "paper_reported_map": 81.42},
        "Pedestrian": {"easy": 57.45, "moderate": 50.42, "hard": 45.36, "paper_reported_map": 51.07},
    },
}


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain(value), indent=2, sort_keys=False), encoding="utf-8")


def json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def configure_determinism(seed: int) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(seed, cuda=torch.cuda.is_available())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    return {
        "seed": seed,
        "amp": False,
        "grad_scaler": "disabled",
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def configure_track(repo: Path, track: str, batch_size: int, epochs: int):
    if track not in TRACK_SPECS:
        raise KeyError(track)
    previous_cwd = Path.cwd()
    try:
        os.chdir(repo / "tools")
        ph_cfg = load_config(Path("cfgs/kitti_models/pointpillar_pillarhist.yaml"))
        pp_cfg = load_config(Path("cfgs/kitti_models/pointpillar.yaml"))
    finally:
        os.chdir(previous_cwd)
    config = configure_r7_track(ph_cfg, pp_cfg, TRACK_SPECS[track]["r7_track"])
    data_path = Path(config.DATA_CONFIG.DATA_PATH)
    if not data_path.is_absolute():
        config.DATA_CONFIG.DATA_PATH = str((repo / "tools" / data_path).resolve())
    config.OPTIMIZATION.BATCH_SIZE_PER_GPU = int(batch_size)
    config.OPTIMIZATION.NUM_EPOCHS = int(epochs)
    if float(config.OPTIMIZATION.LR) != 0.003:
        raise AssertionError("R8 expects the frozen official LR=0.003")
    return config


def build_dataset(config, training: bool, seed: int, batch_size: int):
    from pcdet.datasets import build_dataloader

    set_seed(seed)
    previous_cwd = Path.cwd()
    try:
        os.chdir(Path(__file__).resolve().parents[1])
        dataset, loader, _ = build_dataloader(
            dataset_cfg=config.DATA_CONFIG,
            class_names=config.CLASS_NAMES,
            batch_size=batch_size,
            dist=False,
            workers=0,
            logger=None,
            training=training,
            seed=seed,
        )
    finally:
        os.chdir(previous_cwd)
    return dataset, loader


def make_replay_entries(dataset_size: int, batch_size: int, epochs: int, seed: int):
    if dataset_size <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("dataset_size, batch_size and epochs must be positive")
    global_step = 0
    for epoch in range(epochs):
        order_rng = np.random.default_rng(seed + 17001 + epoch * 100003)
        seed_rng = np.random.default_rng(seed + 29003 + epoch * 100019)
        permutation = order_rng.permutation(dataset_size).tolist()
        sample_seeds = seed_rng.integers(
            1, 2**31 - 1, size=dataset_size, dtype=np.int64
        ).tolist()
        steps = math.ceil(dataset_size / batch_size)
        for batch_index in range(steps):
            start = batch_index * batch_size
            end = min(start + batch_size, dataset_size)
            yield {
                "epoch_one_based": epoch + 1,
                "batch_zero_based": batch_index,
                "global_step_zero_based": global_step,
                "optimizer_step_one_based": global_step + 1,
                "dataset_indices": permutation[start:end],
                "sample_seeds": sample_seeds[start:end],
            }
            global_step += 1


def audit_batch_indices(steps_per_epoch: int, seed: int, epoch_one_based: int):
    if steps_per_epoch < 2:
        return list(range(steps_per_epoch))
    rng = np.random.default_rng(seed + 47017 + epoch_one_based * 100043)
    candidates = np.arange(1, steps_per_epoch - 1)
    random_index = int(rng.choice(candidates)) if candidates.size else 0
    return sorted({0, random_index, steps_per_epoch - 1})


def write_replay(path: Path, dataset_size: int, batch_size: int, epochs: int, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    epoch_permutation_hashes = {}
    audit_batches = {}
    current_epoch = None
    epoch_indices = []
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for entry in make_replay_entries(dataset_size, batch_size, epochs, seed):
            epoch = entry["epoch_one_based"]
            if current_epoch is not None and epoch != current_epoch:
                epoch_permutation_hashes[str(current_epoch)] = hashlib.sha256(
                    json.dumps(epoch_indices, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                epoch_indices = []
            current_epoch = epoch
            epoch_indices.extend(entry["dataset_indices"])
            stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
        if current_epoch is not None:
            epoch_permutation_hashes[str(current_epoch)] = hashlib.sha256(
                json.dumps(epoch_indices, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
    for epoch in range(1, epochs + 1):
        audit_batches[str(epoch)] = audit_batch_indices(steps_per_epoch, seed, epoch)
    return {
        "schema_version": f"{SCHEMA_VERSION}-replay",
        "seed": seed,
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "drop_last": False,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": steps_per_epoch * epochs,
        "generation": {
            "permutation_seed": "seed + 17001 + epoch_zero_based * 100003",
            "augmentation_seed": "seed + 29003 + epoch_zero_based * 100019",
            "numpy_generator": "default_rng/PCG64",
        },
        "epoch_permutation_sha256": epoch_permutation_hashes,
        "audit_batch_indices_zero_based": audit_batches,
        "replay_steps": {"path": str(path), "sha256": sha256_file(path)},
    }


def read_replay(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def update_hash_chain(previous: str, step_id: int, input_checksum: str) -> str:
    payload = bytes.fromhex(previous) + str(step_id).encode("ascii") + input_checksum.encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def db_sampler_state(dataset):
    states = {}
    for index, augmentor in enumerate(dataset.data_augmentor.data_augmentor_queue):
        if hasattr(augmentor, "sample_groups"):
            states[str(index)] = copy.deepcopy(augmentor.sample_groups)
    return states


def restore_db_sampler_state(dataset, states) -> None:
    for index, augmentor in enumerate(dataset.data_augmentor.data_augmentor_queue):
        if str(index) in states:
            augmentor.sample_groups = copy.deepcopy(states[str(index)])


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_canonical_package(model, seed: int):
    state = model.state_dict()
    backend = {
        key: value.detach().cpu().clone()
        for key, value in state.items() if not key.startswith("vfe.")
    }
    return {
        "schema_version": f"{SCHEMA_VERSION}-canonical-init",
        "seed": seed,
        "backend_state": backend,
        "backend_checksum": tensor_dict_checksum(backend),
        "projection_weight": model.vfe.projection.weight.detach().cpu().clone(),
        "projection_bias": model.vfe.projection.bias.detach().cpu().clone(),
    }


def apply_canonical_initialization(model, package, track: str):
    return apply_r7_canonical_initialization(
        model, package, TRACK_SPECS[track]["r7_track"]
    )


def representative_gradients(model):
    backbone_name, backbone_parameter = next(
        (name, parameter)
        for name, parameter in model.backbone_2d.named_parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    )
    selected = {
        f"backbone_2d.{backbone_name}": backbone_parameter.grad,
        "dense_head.conv_cls.weight": model.dense_head.conv_cls.weight.grad,
    }
    if hasattr(model.vfe, "projection"):
        projection = model.vfe.projection
        selected["vfe.projection.weight"] = (
            projection.weight.grad
            if isinstance(projection, torch.nn.Linear)
            else projection[0].weight.grad
        )
    result = {}
    for name, gradient in selected.items():
        result[name] = {
            "present": gradient is not None,
            "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "nonzero": bool(gradient is not None and torch.count_nonzero(gradient).item() > 0),
            "norm": float(gradient.norm().detach().cpu()) if gradient is not None else None,
        }
    result["pass"] = all(item["present"] and item["finite"] and item["nonzero"] for item in result.values())
    return result


def environment_record(repo: Path) -> dict:
    def output(command):
        try:
            return subprocess.check_output(command, cwd=repo, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as error:
            return f"UNAVAILABLE: {error}"

    gpu = output([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    return {
        "created_utc": utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "gpu": gpu,
        "git_head": output(["git", "rev-parse", "HEAD"]),
        "git_branch": output(["git", "branch", "--show-current"]),
    }


class GpuTelemetry:
    FIELDS = ["timestamp", "temperature.gpu", "power.draw", "utilization.gpu", "memory.used", "clocks.gr"]

    def __init__(self, path: Path, interval_seconds: float = 30.0):
        self.path = path
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = None

    def _run(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.FIELDS)
            while not self.stop_event.is_set():
                command = [
                    "nvidia-smi",
                    "--query-gpu=" + ",".join(self.FIELDS),
                    "--format=csv,noheader,nounits",
                ]
                try:
                    row = subprocess.check_output(command, text=True).strip().split(", ")
                except Exception as error:
                    row = [utc_now(), f"ERROR:{error}"] + [""] * (len(self.FIELDS) - 2)
                writer.writerow(row)
                stream.flush()
                self.stop_event.wait(self.interval_seconds)

    def start(self):
        self.thread = threading.Thread(target=self._run, name="r8-gpu-telemetry", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5.0, self.interval_seconds + 1.0))


def recomputed_paper_values():
    result = copy.deepcopy(PAPER_VALUES)
    for model in result.values():
        for category in model.values():
            category["recomputed_from_displayed_ap"] = sum(
                category[key] for key in ("easy", "moderate", "hard")
            ) / 3.0
    return result


def scientific_label(mean_delta_car, mean_delta_ped, positive_car_seeds, positive_ped_seeds):
    values = [mean_delta_car, mean_delta_ped]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("scientific label requires finite unrounded values")
    if (
        mean_delta_car > 0
        and mean_delta_ped > 0
        and positive_car_seeds >= 2
        and positive_ped_seeds >= 2
    ):
        return "DIRECTIONALLY_SUPPORTED"
    if (
        mean_delta_car <= 0
        and mean_delta_ped <= 0
        and positive_car_seeds <= 1
        and positive_ped_seeds <= 1
    ):
        return "NOT_SUPPORTED"
    return "MIXED"
