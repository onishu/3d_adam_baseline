"""Fail-fast helpers for non-finite tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .tensor_stats import tensor_stats


class NonFiniteError(RuntimeError):
    """Raised when a tensor contains NaN/Inf and fail-on-nonfinite is enabled."""


def assert_finite(
    name: str,
    x: Any,
    *,
    enabled: bool = True,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check finiteness; raise NonFiniteError when enabled and violation found."""
    stats = tensor_stats(name, x)
    ok = stats["nan_count"] == 0 and stats["inf_count"] == 0
    if enabled and not ok:
        ctx = context or {}
        raise NonFiniteError(
            f"Non-finite tensor '{name}': nan={stats['nan_count']} inf={stats['inf_count']} "
            f"shape={stats['shape']} context={ctx}"
        )
    return stats


def assert_finite_many(
    items: dict[str, Any],
    *,
    enabled: bool = True,
    context: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {k: assert_finite(k, v, enabled=enabled, context=context) for k, v in items.items()}


def save_debug_bundle(
    out_dir: Path,
    *,
    failure: dict[str, Any],
    input_stats: dict[str, Any],
    loss_stats: dict[str, Any],
    sample_paths: list[str],
    config: dict[str, Any],
    arrays: dict[str, Any] | None = None,
) -> Path:
    """Save NaN debug artifacts. Paths in JSON should be repo-relative when possible."""
    import json
    from datetime import datetime, timezone

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    (out_dir / "input_stats.json").write_text(json.dumps(input_stats, indent=2), encoding="utf-8")
    (out_dir / "loss_stats.json").write_text(json.dumps(loss_stats, indent=2), encoding="utf-8")
    (out_dir / "sample_paths.txt").write_text("\n".join(sample_paths) + "\n", encoding="utf-8")

    try:
        import yaml

        (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    except Exception:
        (out_dir / "config.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if arrays:
        from PIL import Image

        for key, val in arrays.items():
            if val is None:
                continue
            if hasattr(val, "detach"):
                val = val.detach().cpu().numpy()
            arr = np.asarray(val)
            if key.endswith(".npy") or key in {"depth", "depth.npy"}:
                np.save(out_dir / (key if key.endswith(".npy") else f"{key}.npy"), arr)
            elif arr.ndim == 2:
                Image.fromarray(_to_u8(arr)).save(out_dir / f"{key}.png")
            elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                # CHW
                img = arr[:3].transpose(1, 2, 0) if arr.shape[0] >= 3 else arr[0]
                Image.fromarray(_to_u8(img)).save(out_dir / f"{key}.png")
            elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
                Image.fromarray(_to_u8(arr[..., :3] if arr.shape[-1] == 3 else arr[..., 0])).save(
                    out_dir / f"{key}.png"
                )
    meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir).replace("\\", "/"),
    }
    (out_dir / "bundle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def _to_u8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if np.issubdtype(a.dtype, np.floating):
        amin, amax = np.nanmin(a), np.nanmax(a)
        if np.isfinite(amin) and np.isfinite(amax) and amax > amin:
            a = (a - amin) / (amax - amin)
        a = np.clip(a, 0, 1) * 255.0
    return np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)
