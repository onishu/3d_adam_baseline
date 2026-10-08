"""Pytest fixtures for Phase 0–3 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "ADAM3D"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def tiny_adam3d(tmp_path: Path) -> Path:
    """Create a minimal ADAM3D-like category for split/manifest tests."""
    root = tmp_path / "ADAM3D"
    cat = root / "toy"
    # train goods: 5 samples with part/view
    train_rgb = cat / "train" / "good" / "rgb"
    train_xyz = cat / "train" / "good" / "xyz"
    train_rgb.mkdir(parents=True)
    train_xyz.mkdir(parents=True)
    for stem in ["000_A", "000_B", "001_A", "001_B", "002_A"]:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(train_rgb / f"{stem}.png")
        # tiny XYZ tiff via numpy + tifffile if available, else write empty file with .tiff
        xyz = np.zeros((8, 8, 3), dtype=np.float32)
        try:
            import tifffile

            tifffile.imwrite(train_xyz / f"{stem}.tiff", xyz)
        except Exception:
            (train_xyz / f"{stem}.tiff").write_bytes(b"TIFF")

    # disk validation good
    val_rgb = cat / "validation" / "good" / "rgb"
    val_xyz = cat / "validation" / "good" / "xyz"
    val_rgb.mkdir(parents=True)
    val_xyz.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(val_rgb / "003_A.png")
    try:
        import tifffile

        tifffile.imwrite(val_xyz / "003_A.tiff", np.zeros((8, 8, 3), dtype=np.float32))
    except Exception:
        (val_xyz / "003_A.tiff").write_bytes(b"TIFF")

    # defects: one parseable, one numeric-only
    for label, stem, mask_val in [("bulge", "000_A", 255), ("crack", "000", 255)]:
        rgb_dir = cat / "test" / label / "rgb"
        xyz_dir = cat / "test" / label / "xyz"
        gt_dir = cat / "test" / label / "ground_truth"
        rgb_dir.mkdir(parents=True)
        xyz_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(rgb_dir / f"{stem}.png")
        try:
            import tifffile

            tifffile.imwrite(xyz_dir / f"{stem}.tiff", np.zeros((8, 8, 3), dtype=np.float32))
        except Exception:
            (xyz_dir / f"{stem}.tiff").write_bytes(b"TIFF")
        Image.fromarray(np.full((8, 8), mask_val, dtype=np.uint8)).save(gt_dir / f"{stem}.png")

    return root


@pytest.fixture
def dataset_available() -> bool:
    return DATASET_ROOT.is_dir() and (DATASET_ROOT / "1m1").is_dir()
