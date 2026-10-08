"""Atomic TransFusion training / inference checkpoints (no Windows symlinks)."""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


def atomic_torch_save(obj: Any, path: Path) -> None:
    """Write ``obj`` via torch.save then os.replace (atomic on same volume)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def copy_checkpoint(src: Path, dst: Path) -> None:
    """Hard copy (not symlink) for best/last aliases on Windows."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [t.cpu().clone() for t in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        cpu_state = state["torch_cpu"]
        if not isinstance(cpu_state, torch.Tensor):
            cpu_state = torch.tensor(cpu_state, dtype=torch.uint8)
        else:
            cpu_state = cpu_state.cpu().contiguous().to(dtype=torch.uint8)
        torch.set_rng_state(cpu_state)
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = []
        for t in state["torch_cuda"]:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.uint8)
            else:
                t = t.cpu().contiguous().to(dtype=torch.uint8)
            cuda_states.append(t)
        torch.cuda.set_rng_state_all(cuda_states)


def build_checkpoint(
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    config_snapshot: dict[str, Any],
    manifest_sha256: str,
    git_commit: str | None,
    source_repo_commit: str | None,
    dtd_manifest_sha256: str | None,
    foreground_strategy: str,
    best_validation_metric: float | None,
    threshold_from_validation: float | None,
    extras: dict[str, Any] | None = None,
    include_optimizer: bool = True,
) -> dict[str, Any]:
    ckpt: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": (
            optimizer.state_dict() if include_optimizer and optimizer is not None else None
        ),
        "scheduler_state": (
            scheduler.state_dict()
            if include_optimizer and scheduler is not None
            else None
        ),
        "scaler_state": (
            scaler.state_dict() if include_optimizer and scaler is not None else None
        ),
        "rng_state": capture_rng_state() if include_optimizer else None,
        "config_snapshot": config_snapshot,
        "manifest_sha256": manifest_sha256,
        "git_commit": git_commit,
        "source_repo_commit": source_repo_commit,
        "dtd_manifest_sha256": dtd_manifest_sha256,
        "foreground_strategy": foreground_strategy,
        "best_validation_metric": best_validation_metric,
        "threshold_selected_from_validation": threshold_from_validation,
        "checkpoint_kind": "training" if include_optimizer else "inference",
        "extras": extras or {},
    }
    return ckpt


def build_inference_checkpoint(
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    config_snapshot: dict[str, Any],
    manifest_sha256: str,
    git_commit: str | None,
    source_repo_commit: str | None,
    validation_metric_name: str,
    validation_metric_value: float | None,
    foreground_strategy: str,
    threshold_from_validation: float | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_kind": "inference",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "config_snapshot": config_snapshot,
        "manifest_sha256": manifest_sha256,
        "git_commit": git_commit,
        "source_repo_commit": source_repo_commit,
        "foreground_strategy": foreground_strategy,
        "validation_metric_name": validation_metric_name,
        "validation_metric_value": validation_metric_value,
        "threshold_selected_from_validation": threshold_from_validation,
        "extras": extras or {},
    }


def prune_periodic_checkpoints(ckpt_dir: Path, keep: int) -> list[str]:
    """Keep newest ``keep`` epoch_XXXX.ckpt files; never touch best/last/inference."""
    if keep < 0:
        return []
    ckpt_dir = Path(ckpt_dir)
    epoch_files = sorted(ckpt_dir.glob("epoch_*.ckpt"))
    removed: list[str] = []
    if keep == 0:
        victims = epoch_files
    else:
        victims = epoch_files[:-keep] if len(epoch_files) > keep else []
    for path in victims:
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def save_training_checkpoint(
    ckpt: dict[str, Any],
    ckpt_dir: Path,
    *,
    is_best: bool = False,
    write_epoch_file: bool = True,
    update_last: bool = True,
    keep_periodic: int | None = None,
) -> Path:
    """Save epoch_XXXX.ckpt plus last.ckpt (and best.ckpt if requested)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(ckpt["epoch"])
    epoch_path = ckpt_dir / f"epoch_{epoch:04d}.ckpt"
    if write_epoch_file:
        atomic_torch_save(ckpt, epoch_path)
        target = epoch_path
    else:
        # Still materialize last without keeping an epoch_ file.
        target = ckpt_dir / "last.ckpt"
        atomic_torch_save(ckpt, target)
    if update_last and write_epoch_file:
        copy_checkpoint(epoch_path, ckpt_dir / "last.ckpt")
    if is_best and write_epoch_file:
        copy_checkpoint(epoch_path, ckpt_dir / "best.ckpt")
    if keep_periodic is not None and write_epoch_file:
        prune_periodic_checkpoints(ckpt_dir, keep_periodic)
    return target


def save_inference_checkpoint(ckpt: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    atomic_torch_save(ckpt, path)
    return path


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    state = ckpt.get("model_state") or ckpt.get("net_state_dict") or ckpt
    if isinstance(state, dict) and any(k.startswith("model.") or k.startswith("unet") for k in state):
        model.load_state_dict(state)
    else:
        model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if restore_rng:
        restore_rng_state(ckpt.get("rng_state"))
    return ckpt
