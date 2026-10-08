"""Compare TransFusion foreground strategies on 1m1 train samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from scipy import ndimage

from src.common.config import DEFAULT_DATASET_ROOT, REPO_ROOT
from src.common.result_schema import atomic_write_json
from src.data.adapters import transfusion as tf_adapter


def _connected_components(mask: np.ndarray) -> int:
    labeled, n = ndimage.label(mask.astype(bool))
    return int(n)


def audit_category(root: Path, category: str, thr: float = 0.005) -> list[dict[str, Any]]:
    xyz_dir = root / category / "train" / "good" / "xyz"
    rows: list[dict[str, Any]] = []
    tf_adapter.ensure_tf_on_path(chdir=False)

    for path in sorted(xyz_dir.glob("*.tiff")):
        arr = tifffile.imread(str(path)).astype(np.float32)
        arr = cv2.resize(arr, (320, 256), interpolation=cv2.INTER_NEAREST)
        valid = arr[:, :, 2] != 0
        valid_count = int(valid.sum())

        strategies = (
            "upstream_corner_fit",
            "valid_point_plane_fit",
            "valid_point_plane_fit_with_depth_fallback",
            "valid_depth_mask",
            "provided_foreground_mask",
            "adam3d_valid_point_fit",
        )
        for strategy in strategies:
            meta = {"degenerate": False, "fallback_used": False}
            if strategy in {
                "adam3d_valid_point_fit",
                "valid_point_plane_fit_with_depth_fallback",
            }:
                mask, meta = tf_adapter.get_plane_mask_adam3d(
                    arr, thr=thr, return_meta=True, allow_fallback=True
                )
            elif strategy == "valid_point_plane_fit":
                try:
                    mask, meta = tf_adapter.get_plane_mask_adam3d(
                        arr, thr=thr, return_meta=True, allow_fallback=False
                    )
                except RuntimeError as exc:
                    mask = np.zeros((*valid.shape, 1), dtype=np.float32)
                    meta = {
                        "degenerate": True,
                        "fallback_used": False,
                        "failure_reason": str(exc),
                    }
            elif strategy == "upstream_corner_fit":
                with np.errstate(divide="ignore", invalid="ignore"):
                    mask = tf_adapter.get_plane_mask_upstream_corner(arr, thr=thr)
                if not np.isfinite(mask).all():
                    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
                    meta["degenerate"] = True
            elif strategy in {"valid_depth_mask", "provided_foreground_mask"}:
                mask = (valid.astype(np.float32))[:, :, None]
            else:
                raise RuntimeError(strategy)
            fg = mask.squeeze() > 0.5
            rows.append(
                {
                    "sample": path.stem,
                    "strategy": strategy,
                    "valid_depth_pixels": valid_count,
                    "foreground_pixels": int(fg.sum()),
                    "foreground_ratio": float(fg.mean()) if fg.size else 0.0,
                    "connected_components": _connected_components(fg),
                    "degenerate": bool(meta.get("degenerate", False)),
                    "fallback_used": bool(meta.get("fallback_used", False)),
                    "all_zero": int(fg.sum()) == 0,
                }
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--category", default="1m1")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=REPO_ROOT / "results" / "quality" / "transfusion_foreground_1m1.csv",
    )
    p.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT / "results" / "quality" / "transfusion_foreground_summary.json",
    )
    args = p.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    rows = audit_category(root, args.category)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    summary: dict[str, Any] = {"category": args.category, "n_samples": 0, "by_strategy": {}}
    by_sample = {r["sample"] for r in rows}
    summary["n_samples"] = len(by_sample)
    strategies = sorted({r["strategy"] for r in rows})
    for strategy in strategies:
        sub = [r for r in rows if r["strategy"] == strategy]
        n = len(sub)
        zero = sum(1 for r in sub if r["all_zero"])
        summary["by_strategy"][strategy] = {
            "n": n,
            "all_zero_count": zero,
            "all_zero_rate": (zero / n) if n else None,
            "mean_foreground_ratio": float(np.mean([r["foreground_ratio"] for r in sub])) if sub else None,
            "fallback_rate": float(np.mean([r["fallback_used"] for r in sub])) if sub else None,
        }
    atomic_write_json(args.out_summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
