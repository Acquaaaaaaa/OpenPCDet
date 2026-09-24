"""Assemble immutable provenance files for one Booth A4 experiment directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_exact(source: Path, destination: Path) -> None:
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise FileExistsError(f"existing provenance file differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--bootstrap-dir", type=Path, required=True)
    parser.add_argument("--frame-lists-dir", type=Path, required=True)
    args = parser.parse_args()
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    _copy_exact(args.resolved_config, args.experiment_dir / "resolved_config.yaml")
    _copy_exact(args.bootstrap_dir / "baseline_checks.json", args.experiment_dir / "baseline_checks.json")
    _copy_exact(args.bootstrap_dir / "activation_registry.json", args.experiment_dir / "activation_registry.json")
    for source in sorted(args.frame_lists_dir.glob("*")):
        if source.is_file():
            _copy_exact(source, args.experiment_dir / "frame_lists" / source.name)

    environment = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.experiment_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    manifest = {
        "experiment_id": args.experiment_dir.name,
        "conclusion_scope": "fp32_reference_offline_quantized_int8",
        "git_commit": git_commit,
        "git_status_porcelain": git_status,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "pointpillar_config": str(args.cfg_file.resolve()),
        "pointpillar_config_sha256": sha256_file(args.cfg_file),
        "resolved_config_sha256": sha256_file(args.resolved_config),
        "seed": 666,
    }
    manifest_path = args.experiment_dir / "manifest.json"
    serialized = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if manifest_path.exists() and manifest_path.read_bytes() != serialized:
        raise FileExistsError("existing experiment manifest differs")
    manifest_path.write_bytes(serialized)
    print(json.dumps({"experiment_id": manifest["experiment_id"], "git_commit": git_commit}))


if __name__ == "__main__":
    main()
