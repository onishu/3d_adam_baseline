"""CPU tests for TransFusion ADAM3D adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common.tensor_stats import stats_are_json_serializable, tensor_stats
from src.data.adapters.transfusion import (
    get_plane_mask_adam3d,
    read_rgb_depth_mask,
    sample_raw_paths,
    validate_dtd,
)


def test_tensor_stats_json_serializable() -> None:
    s = tensor_stats("x", np.array([0.0, 1.0, np.nan]))
    assert s["nan_count"] == 1
    assert stats_are_json_serializable(s)


def test_validate_dtd_missing(tmp_path: Path) -> None:
    info = validate_dtd(tmp_path / "missing", min_images=10)
    assert info["ok"] is False


def test_plane_mask_adam3d_no_corner_divzero() -> None:
    h, w = 32, 40
    xyz = np.zeros((h, w, 3), dtype=np.float32)
    # corners remain zero; object in center
    xyz[8:24, 8:32, :] = 0.4
    xyz[8:24, 8:32, 2] = 0.6
    m = get_plane_mask_adam3d(xyz, thr=0.005)
    assert m.shape == (h, w, 1)
    assert np.isfinite(m).all()
    assert float(m.sum()) > 0


def test_plane_mask_rejects_empty_valid() -> None:
    xyz = np.zeros((16, 16, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="insufficient valid depth"):
        get_plane_mask_adam3d(xyz)


def test_windows_path_and_pairing(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb" / "001_B.png"
    xyz = tmp_path / "xyz" / "001_B.tiff"
    rgb.parent.mkdir(); xyz.parent.mkdir()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(rgb)
    arr = np.zeros((16, 16, 3), dtype=np.float32)
    arr[4:12, 4:12, 2] = 0.5
    import tifffile

    tifffile.imwrite(str(xyz), arr)
    out = read_rgb_depth_mask(rgb, xyz, None, width=32, height=24)
    assert out["rgb_chw"].ndim == 3
    # path stem pairing
    assert rgb.stem == xyz.stem


@pytest.mark.dataset
def test_load_one_real_sample_and_shapes() -> None:
    root = Path("datasets/ADAM3D")
    if not (root / "1m1").is_dir():
        pytest.skip("ADAM3D missing")
    paths = sample_raw_paths(root, "1m1", split="train")
    out = read_rgb_depth_mask(paths["image_path"], paths["depth_path"], None)
    assert out["rgb_chw"].shape[0] == 3
    assert np.isfinite(out["xyz_hwc"]).all()
    # spatial agreement is via upstream convention docs; both defined
    assert out["rgb_spatial"][0] == 320
    assert out["xyz_spatial"][0] == 256


@pytest.mark.dataset
def test_manifest_can_resolve_same_sample() -> None:
    import csv

    root = Path("datasets/ADAM3D")
    man = Path("configs/splits/anomalib_manifest.csv")
    if not man.is_file() or not (root / "1m1").is_dir():
        pytest.skip("manifest/dataset missing")
    with man.open(encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["category"] == "1m1" and r["split"] == "train")
    img = root / row["image_path"]
    depth = root / row["depth_path"]
    assert img.is_file() and depth.is_file()
    assert img.stem == depth.stem
