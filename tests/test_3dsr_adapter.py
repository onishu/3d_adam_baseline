"""CPU tests for 3DSR ADAM3D adapter helpers (no GPU / no training)."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import REPO_ROOT
from src.data.adapters import dsr

DATASET = REPO_ROOT / "datasets" / "ADAM3D"


@pytest.mark.skipif(not DATASET.is_dir(), reason="ADAM3D dataset missing")
def test_read_xyz_z_shape_and_channels_equal_flag():
    paths = dsr.sample_paths(DATASET, "1m1", "train")
    out = dsr.read_xyz_z(paths["xyz"], size=384)
    assert out["spatial"] == (384, 384)
    assert out["valid_ratio"] > 0
    # anomalib subset stores X=Y=Z
    assert out["channels_equal_xyz"] is True


@pytest.mark.skipif(not DATASET.is_dir(), reason="ADAM3D dataset missing")
def test_normalize_depth_band():
    paths = dsr.sample_paths(DATASET, "1m1", "train")
    out = dsr.read_xyz_z(paths["xyz"], size=64)
    norm = dsr.normalize_depth_author_style(out["z"], out["valid_mask"] > 0.5)
    valid = out["valid_mask"] > 0.5
    assert np.isfinite(norm).all()
    assert float(norm[valid].min()) >= 0.1 - 1e-5
    assert float(norm[valid].max()) <= 0.9 + 1e-5
    assert float(norm[~valid].max()) == 0.0


@pytest.mark.skipif(not DATASET.is_dir(), reason="ADAM3D dataset missing")
def test_read_rgb_chw():
    paths = dsr.sample_paths(DATASET, "1m1", "train")
    rgb = dsr.read_rgb(paths["rgb"], size=384)
    assert rgb.shape == (3, 384, 384)
    assert 0.0 <= float(rgb.min()) and float(rgb.max()) <= 1.0


def test_pretrained_vq_listing_structure():
    items = dsr.list_pretrained_vq()
    names = {i["name"] for i in items}
    assert "DADA_D.pckl" in names
    assert "DADA_RGB_D.pckl" in names
