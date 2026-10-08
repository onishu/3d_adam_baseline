"""ADAM3D adapter helpers for VitjanZ/3DSR (CPU-side; no training here)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from src.common.config import REPO_ROOT
from src.common.finite_check import assert_finite

UPSTREAM_3DSR = {
    "url": "https://github.com/VitjanZ/3DSR",
    "commit": "6f83317837c6615f893f3107235c75edda90181a",
    "license": "Apache-2.0",
}

DSR_ROOT = REPO_ROOT / "third_party" / "3DSR"
TARGET_SIZE = 384  # author img_dim


def dsr_available() -> bool:
    return DSR_ROOT.is_dir() and (DSR_ROOT / "data_loader.py").is_file()


def list_pretrained_vq() -> list[dict[str, Any]]:
    """Expected upstream VQ checkpoints (may be absent until download)."""
    names = [
        ("DADA_D.pckl", "depth", "train_dsr_depth.py"),
        ("DADA_RGB_D.pckl", "rgbd", "train_dsr.py"),
        ("DADA_dtest_.pckl", "depth_experimental", None),
        ("DADA_l1_.pckl", "rgbd_experimental", None),
    ]
    out = []
    for name, modality, used_by in names:
        path = DSR_ROOT / "checkpoints" / name
        out.append(
            {
                "name": name,
                "modality": modality,
                "used_by_default": used_by,
                "path": str(path).replace("\\", "/"),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return out


def read_xyz_z(
    path: Path,
    *,
    size: int = TARGET_SIZE,
) -> dict[str, Any]:
    """Load XYZ TIFF, take Z, resize like 3DSR (cv2 dsize=(W,H))."""
    xyz = tifffile.imread(str(path)).astype(np.float32)
    if xyz.ndim != 3 or xyz.shape[-1] < 3:
        raise ValueError(f"Expected HxWx3 XYZ, got {xyz.shape}")
    channels_equal = bool(
        np.allclose(xyz[:, :, 0], xyz[:, :, 1]) and np.allclose(xyz[:, :, 0], xyz[:, :, 2])
    )
    xyz_r = cv2.resize(xyz, (size, size), interpolation=cv2.INTER_NEAREST)
    z = xyz_r[:, :, 2]
    valid = z != 0
    assert_finite("xyz_z", z, enabled=True)
    return {
        "z": z,
        "valid_mask": valid.astype(np.float32),
        "spatial": tuple(z.shape),
        "channels_equal_xyz": channels_equal,
        "valid_ratio": float(valid.mean()),
    }


def normalize_depth_author_style(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Match data_loader band-normalize + zero invalid (without fill_depth_map)."""
    z = z.astype(np.float32).copy()
    inv = ~valid.astype(bool)
    if (~inv).sum() == 0:
        raise RuntimeError("no valid depth pixels")
    # author: im_min from valid pixels only (invalid contribution forced high)
    z_masked = z * (1.0 - inv.astype(np.float32))
    im_min = float(np.min(z_masked + 1000.0 * inv.astype(np.float32)))
    im_max = float(np.max(z_masked))
    out = (z - im_min) / max(im_max - im_min, 1e-12)
    out = out * 0.8 + 0.1
    out = out * (1.0 - inv.astype(np.float32))
    return out.astype(np.float32)


def read_rgb(path: Path, *, size: int = TARGET_SIZE) -> np.ndarray:
    """BGR file → RGB float [0,1], CHW."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


def sample_paths(dataset_root: Path, category: str, split: str = "train") -> dict[str, Path]:
    if split == "train":
        rgb_dir = dataset_root / category / "train" / "good" / "rgb"
        rgb = sorted(rgb_dir.glob("*.png"))[0]
        xyz = dataset_root / category / "train" / "good" / "xyz" / f"{rgb.stem}.tiff"
        return {"rgb": rgb, "xyz": xyz, "mask": Path()}
    test_root = dataset_root / category / "test"
    for label_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        rgb_dir = label_dir / "rgb"
        if not rgb_dir.is_dir():
            continue
        rgb = sorted(rgb_dir.glob("*.png"))[0]
        xyz = label_dir / "xyz" / f"{rgb.stem}.tiff"
        mask = label_dir / "ground_truth" / f"{rgb.stem}.png"
        return {"rgb": rgb, "xyz": xyz, "mask": mask}
    raise FileNotFoundError(test_root)
