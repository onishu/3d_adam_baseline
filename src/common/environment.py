"""Capture runtime / package / git environment metadata."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .hashing import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]


def _safe_import_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def git_commit(repo: Path | None = None) -> str | None:
    root = repo or REPO_ROOT
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def git_dirty(repo: Path | None = None) -> bool | None:
    root = repo or REPO_ROOT
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return None


def capture_environment(include_freeze: bool = False) -> dict[str, Any]:
    """Collect environment fields used by experiment result JSON."""
    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "git_commit": git_commit(),
        "dirty_worktree": git_dirty(),
        "package_versions": {
            "anomalib": _safe_import_version("anomalib"),
            "torch": _safe_import_version("torch"),
            "torchvision": _safe_import_version("torchvision"),
            "pandas": _safe_import_version("pandas"),
            "numpy": _safe_import_version("numpy"),
            "sklearn": _safe_import_version("sklearn"),
        },
        "cuda_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram": None,
    }

    lock = REPO_ROOT / "uv.lock"
    if lock.is_file():
        info["uv_lock_sha256"] = sha256_file(lock)

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram"] = int(props.total_memory)
            info["gpu_vram_gb"] = round(props.total_memory / (1024**3), 3)
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"

    if include_freeze:
        try:
            out = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            info["package_freeze"] = out.splitlines()
        except Exception as exc:
            info["package_freeze_error"] = f"{type(exc).__name__}: {exc}"

    return info
