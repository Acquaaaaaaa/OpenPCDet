"""Freeze baseline evidence and validate the PointPillars activation registry."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.detectors.detector3d_template import _load_checkpoint
from pcdet.utils import common_utils
from tools.booth_a4.activation_registry import (
    ActivationShapeObserver,
    build_activation_registry,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*command: str) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def outputs_equal(reference, observed) -> bool:
    if isinstance(reference, torch.Tensor):
        return (
            isinstance(observed, torch.Tensor)
            and reference.shape == observed.shape
            and reference.dtype == observed.dtype
            and torch.equal(reference, observed)
        )
    if isinstance(reference, dict):
        return (
            isinstance(observed, dict)
            and reference.keys() == observed.keys()
            and all(outputs_equal(reference[key], observed[key]) for key in reference)
        )
    if isinstance(reference, (list, tuple)):
        return (
            isinstance(observed, type(reference))
            and len(reference) == len(observed)
            and all(outputs_equal(left, right) for left, right in zip(reference, observed))
        )
    if isinstance(reference, np.ndarray):
        return isinstance(observed, np.ndarray) and np.array_equal(reference, observed)
    return reference == observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cfg_file = args.cfg_file.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    # OpenPCDet's legacy _BASE_CONFIG_ and DATA_PATH values are relative to tools/.
    os.chdir(REPO_ROOT / "tools")
    cfg_from_yaml_file(str(args.cfg_file), cfg)
    cfg.TAG = args.cfg_file.stem
    np.random.seed(666)
    torch.manual_seed(666)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(666)
    logger = common_utils.create_logger(args.output_dir / "bootstrap.log", rank=0)
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )
    model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)

    checkpoint = _load_checkpoint(str(args.checkpoint), map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    registry = build_activation_registry(model)
    registry.write(args.output_dir)

    if len(registry.consumers) != 23 or len(registry.edges) != 19:
        raise AssertionError(
            f"unexpected registry cardinality: {len(registry.consumers)} consumers, "
            f"{len(registry.edges)} unique activations"
        )

    device = torch.device(args.device)
    model.to(device).eval()
    raw_batch = next(iter(loader))
    batch_without_hooks = copy.deepcopy(raw_batch)
    batch_with_hooks = copy.deepcopy(raw_batch)
    load_data_to_gpu(batch_without_hooks) if device.type == "cuda" else None
    load_data_to_gpu(batch_with_hooks) if device.type == "cuda" else None
    if device.type == "cpu":
        for batch in (batch_without_hooks, batch_with_hooks):
            for key, value in list(batch.items()):
                if isinstance(value, np.ndarray) and key not in {
                    "frame_id", "metadata", "calib", "image_paths", "ori_shape", "img_process_infos"
                }:
                    batch[key] = torch.from_numpy(value).float()

    with torch.inference_mode():
        reference_output = model(batch_without_hooks)
        observer = ActivationShapeObserver(model, registry)
        with observer:
            observed_output = model(batch_with_hooks)
    observer.validate()
    output_unchanged = outputs_equal(reference_output, observed_output)
    if not output_unchanged:
        raise AssertionError("model outputs changed after installing non-mutating pre-hooks")
    (args.output_dir / "sample_observations.json").write_text(
        json.dumps(observer.observations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checks = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_porcelain": subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size": args.checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config": str(args.cfg_file.resolve()),
        "config_sha256": sha256_file(args.cfg_file),
        "strict_checkpoint_load": True,
        "consumer_count": len(registry.consumers),
        "unique_activation_count": len(registry.edges),
        "sample_frame_ids": list(raw_batch["frame_id"]),
        "hooks_preserved_outputs_exactly": output_unchanged,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "seed": 666,
    }
    (args.output_dir / "baseline_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: checks[key] for key in (
        "consumer_count", "unique_activation_count", "hooks_preserved_outputs_exactly"
    )}))


if __name__ == "__main__":
    main()
