"""Export ADAM3D organized XYZ TIFFs to Real3D-AD-style point clouds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from .categories import CATEGORIES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="datasets/ADAM3D")
    p.add_argument("--out", default="datasets/ADAM3D_pcd")
    p.add_argument("--categories", nargs="*")
    p.add_argument("--max-points", type=int, default=0, help="0 keeps all valid points")
    return p.parse_args()


def organized_xyz(tiff_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    arr = tifffile.imread(str(tiff_path)).astype(np.float32)
    if arr.ndim == 2:
        h, w = arr.shape
        ys, xs = np.mgrid[0:h, 0:w]
        pts = np.stack([xs, ys, arr], axis=-1).reshape(-1, 3)
    else:
        h, w = arr.shape[:2]
        pts = arr.reshape(-1, arr.shape[-1])[:, :3]
    return pts, (h, w)


def valid_mask(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 0)


def sample_points(points: np.ndarray, labels: np.ndarray | None, max_points: int):
    if max_points <= 0 or len(points) <= max_points:
        return points, labels
    idx = np.random.choice(len(points), max_points, replace=False)
    if labels is None:
        return points[idx], None
    return points[idx], labels[idx]


def write_pcd(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def write_gt_txt(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.concatenate([points, labels.reshape(-1, 1)], axis=1)
    np.savetxt(path, data, fmt="%.6f")


def export_category(root: Path, out: Path, category: str, max_points: int) -> tuple[int, int]:
    cat_in = root / category
    cat_out = out / category
    n_train = n_test = 0

    for rgb in sorted((cat_in / "train" / "good" / "rgb").glob("*.png")):
        xyz = cat_in / "train" / "good" / "xyz" / f"{rgb.stem}.tiff"
        points, _ = organized_xyz(xyz)
        points = points[valid_mask(points)]
        points, _ = sample_points(points, None, max_points)
        write_pcd(cat_out / "train" / f"{rgb.stem}_template.pcd", points)
        n_train += 1

    test_root = cat_in / "test"
    for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        defect = defect_dir.name
        for rgb in sorted((defect_dir / "rgb").glob("*.png")):
            xyz = defect_dir / "xyz" / f"{rgb.stem}.tiff"
            points, (h, w) = organized_xyz(xyz)
            valid = valid_mask(points)
            points = points[valid]
            mask_path = defect_dir / "ground_truth" / f"{rgb.stem}.png"
            if defect == "good" or not mask_path.exists():
                points, _ = sample_points(points, None, max_points)
                write_pcd(cat_out / "test" / f"good_{rgb.stem}.pcd", points)
            else:
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask.shape != (h, w):
                    mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))
                labels = (mask.reshape(-1) > 0).astype(np.float32)[valid]
                points, labels = sample_points(points, labels, max_points)
                stem = f"{defect}_{rgb.stem}"
                write_pcd(cat_out / "test" / f"{stem}.pcd", points)
                write_gt_txt(cat_out / "gt" / f"{stem}.txt", points, labels)
            n_test += 1
    return n_train, n_test


def main():
    args = parse_args()
    categories = args.categories or CATEGORIES
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for category in categories:
        n_train, n_test = export_category(root, out, category, args.max_points)
        print(f"{category}: train={n_train} test={n_test}")
    print("Exported PCD dataset:", out)


if __name__ == "__main__":
    main()
