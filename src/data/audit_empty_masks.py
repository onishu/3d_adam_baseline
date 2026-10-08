"""Audit empty ground-truth masks in ADAM3D anomalib subset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from src.categories import CATEGORIES
from src.common.config import DEFAULT_DATASET_ROOT, REPO_ROOT


def classify(label: str, pixel_sum: int, mask_exists: bool) -> str:
    if not mask_exists:
        return "missing_annotation"
    if label == "good":
        return "expected_empty_normal" if pixel_sum == 0 else "unresolved"
    if pixel_sum == 0:
        # Defect folder but empty mask: cannot tell visibility without expert review.
        return "unresolved"
    return "unresolved"


def audit(root: Path, categories: list[str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for category in categories:
        test_root = root / category / "test"
        if not test_root.is_dir():
            continue
        for label_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
            anomaly_type = label_dir.name
            rgb_dir = label_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            for rgb in sorted(rgb_dir.glob("*.png")):
                mask_path = label_dir / "ground_truth" / f"{rgb.stem}.png"
                depth_path = label_dir / "xyz" / f"{rgb.stem}.tiff"
                mask_exists = mask_path.is_file()
                pixel_sum = -1
                if mask_exists:
                    m = np.array(Image.open(mask_path).convert("L"))
                    pixel_sum = int(m.sum())
                depth_valid = -1
                if depth_path.is_file():
                    xyz = tifffile.imread(str(depth_path))
                    if xyz.ndim == 3:
                        z = xyz[:, :, 2] if xyz.shape[-1] == 3 else xyz[2]
                    else:
                        z = xyz
                    depth_valid = int(np.isfinite(z).sum() - (z == 0).sum()) if np.issubdtype(z.dtype, np.floating) else int((z != 0).sum())
                label = "good" if anomaly_type == "good" else "abnormal"
                # Only keep empty masks for abnormal (and missing) for the 28-count report,
                # but also list good empties separately if requested.
                if label == "abnormal" and mask_exists and pixel_sum == 0:
                    classification = "unresolved"
                elif label == "abnormal" and not mask_exists:
                    classification = "missing_annotation"
                elif label == "good" and pixel_sum == 0:
                    classification = "expected_empty_normal"
                else:
                    continue
                # Refine unresolved empty abnormal masks
                if classification == "unresolved" and pixel_sum == 0:
                    # Heuristic: if depth has almost no valid points, likely conversion_error
                    if depth_valid == 0:
                        classification = "conversion_error"
                    else:
                        classification = "anomalous_but_not_visible"
                rows.append(
                    {
                        "category": category,
                        "image_path": str(rgb.relative_to(root)).replace("\\", "/"),
                        "label": label,
                        "anomaly_type": anomaly_type,
                        "mask_path": str(mask_path.relative_to(root)).replace("\\", "/") if mask_exists else "",
                        "mask_exists": mask_exists,
                        "mask_pixel_sum": pixel_sum,
                        "depth_valid_pixel_count": depth_valid,
                        "classification": classification,
                    }
                )
    summary = {
        "n_rows": len(rows),
        "by_classification": dict(Counter(r["classification"] for r in rows)),
        "by_category": dict(Counter(r["category"] for r in rows)),
        "note": (
            "Empty abnormal masks cannot be confirmed as missing_annotation vs "
            "anomalous_but_not_visible without expert review; default heuristic uses "
            "anomalous_but_not_visible when depth has valid points."
        ),
    }
    return rows, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--categories", nargs="*")
    p.add_argument("--out-csv", type=Path, default=REPO_ROOT / "results" / "quality" / "empty_masks.csv")
    p.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT / "results" / "quality" / "empty_masks_summary.json",
    )
    a = p.parse_args()
    cats = a.categories or CATEGORIES
    rows, summary = audit(Path(a.root), cats)
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "image_path",
                "label",
                "anomaly_type",
                "mask_path",
                "mask_exists",
                "mask_pixel_sum",
                "depth_valid_pixel_count",
                "classification",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    a.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"wrote_csv": str(a.out_csv), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
