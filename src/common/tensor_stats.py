"""Collect finite/NaN/Inf statistics for tensors and numpy arrays."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if x is None:
        raise ValueError("Cannot compute stats for None")
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    arr = np.asarray(x)
    if arr.dtype == object:
        raise TypeError(f"Unsupported dtype for stats: {arr.dtype}")
    return arr.astype(np.float64, copy=False)


def tensor_stats(name: str, x: Any) -> dict[str, Any]:
    """Return JSON-serializable stats for a tensor/array."""
    arr = _to_numpy(x)
    flat = arr.reshape(-1)
    n = int(flat.size)
    if n == 0:
        return {
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(getattr(x, "dtype", arr.dtype)),
            "count": 0,
            "nan_count": 0,
            "inf_count": 0,
            "finite_rate": None,
            "zero_rate": None,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    nan_count = int(np.isnan(flat).sum())
    inf_count = int(np.isinf(flat).sum())
    finite = flat[np.isfinite(flat)]
    finite_rate = float(finite.size / n)
    zero_rate = float((flat == 0).sum() / n)
    out: dict[str, Any] = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(getattr(x, "dtype", arr.dtype)),
        "count": n,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_rate": finite_rate,
        "zero_rate": zero_rate,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
    }
    return out


def stats_are_json_serializable(stats: dict[str, Any]) -> bool:
    import json

    try:
        json.dumps(stats)
        return True
    except TypeError:
        return False
