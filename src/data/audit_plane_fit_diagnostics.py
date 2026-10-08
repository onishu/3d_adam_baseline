"""Per-sample plane-fit diagnostics for TransFusion foreground strategies."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from src.common.config import DEFAULT_DATASET_ROOT, REPO_ROOT
from src.common.result_schema import atomic_write_json
from src.data.adapters import transfusion as tf_adapter


def audit(root: Path, category: str, thr: float = 0.005) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    xyz_dir = root / category / "train" / "good" / "xyz"
    rows: list[dict[str, Any]] = []
    tf_adapter.ensure_tf_on_path(chdir=False)
    reasons: Counter[str] = Counter()
    for path in sorted(xyz_dir.glob("*.tiff")):
        arr = tifffile.imread(str(path)).astype(np.float32)
        arr = cv2.resize(arr, (320, 256), interpolation=cv2.INTER_NEAREST)
        diag = tf_adapter.diagnose_plane_fit(arr, thr=thr)
        reason = diag.get("failure_reason") or "ok"
        reasons[reason] += 1
        rows.append(
            {
                "sample": path.stem,
                "valid_point_count": diag.get("valid_point_count"),
                "sampled_point_count": diag.get("sampled_point_count"),
                "coordinate_min": json.dumps(diag.get("coordinate_min")),
                "coordinate_max": json.dumps(diag.get("coordinate_max")),
                "coordinate_scale": diag.get("coordinate_scale"),
                "matrix_rank": diag.get("matrix_rank"),
                "singular_values": json.dumps(diag.get("singular_values")),
                "estimated_normal": json.dumps(diag.get("estimated_normal")),
                "plane_offset": diag.get("plane_offset"),
                "residual_min": diag.get("residual_min"),
                "residual_mean": diag.get("residual_mean"),
                "residual_max": diag.get("residual_max"),
                "inlier_threshold": diag.get("inlier_threshold"),
                "inlier_count": diag.get("inlier_count"),
                "foreground_count": diag.get("foreground_count"),
                "failure_reason": diag.get("failure_reason"),
                "fallback_used": diag.get("fallback_used"),
            }
        )
    summary = {
        "category": category,
        "n_samples": len(rows),
        "failure_reason_counts": dict(reasons),
        "fallback_rate": float(sum(1 for r in rows if r["fallback_used"]) / max(len(rows), 1)),
        "note": (
            "anomalib-subset XYZ TIFFs store identical values in X/Y/Z channels "
            "(true organized XYZ is unavailable). Sampled points are collinear "
            "(matrix_rank=1), get_plane_from_points returns a zero normal, and "
            "plane fit fails with degenerate_plane_coefficients on every sample. "
            "valid_point_plane_fit_with_depth_fallback therefore falls back 100% "
            "to valid_depth_mask. Pilot A/B should use valid_depth_mask explicitly."
        ),
    }
    return rows, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--category", default="1m1")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=REPO_ROOT / "results" / "quality" / "transfusion_plane_fit_diagnostics_1m1.csv",
    )
    p.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "quality"
        / "transfusion_plane_fit_diagnostics_summary.json",
    )
    args = p.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    rows, summary = audit(root, args.category)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    atomic_write_json(args.out_summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
