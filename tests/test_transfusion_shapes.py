"""Shape / orientation tests for TransFusion ADAM3D adapter (CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.adapters.transfusion import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    expect_upstream_tensor_spatial,
    read_rgb_depth_mask,
)


@pytest.fixture
def tiny_sample(tmp_path: Path) -> dict[str, Path]:
    rgb = tmp_path / "000_A.png"
    xyz = tmp_path / "000_A.tiff"
    mask = tmp_path / "000_A_mask.png"
    Image.fromarray(np.zeros((64, 80, 3), dtype=np.uint8)).save(rgb)
    Image.fromarray(np.zeros((64, 80), dtype=np.uint8)).save(mask)
    arr = np.zeros((64, 80, 3), dtype=np.float32)
    arr[10:50, 10:70, :] = 0.5
    arr[10:50, 10:70, 2] = 0.7
    try:
        import tifffile

        tifffile.imwrite(str(xyz), arr)
    except Exception:
        pytest.skip("tifffile unavailable")
    return {"image_path": rgb, "depth_path": xyz, "mask_path": mask}


def test_expect_upstream_spatial_convention() -> None:
    # CLI width=320,height=256 -> upstream tensor (H,W)=(320,256)
    assert expect_upstream_tensor_spatial(320, 256) == (320, 256)


def test_rgb_depth_spatial_match_upstream_convention(tiny_sample: dict[str, Path]) -> None:
    out = read_rgb_depth_mask(
        tiny_sample["image_path"],
        tiny_sample["depth_path"],
        tiny_sample["mask_path"],
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
        use_upstream_resize_convention=True,
    )
    assert out["rgb_chw"].shape == (3, TARGET_WIDTH, TARGET_HEIGHT)
    # cv2.resize dsize=(width,height)=(320,256) => array HxW = (256,320)
    assert out["xyz_hwc"].shape[:2] == (TARGET_HEIGHT, TARGET_WIDTH)
    assert out["mask_hw"].shape == (TARGET_WIDTH, TARGET_HEIGHT)


def test_mask_nearest_preserves_binary(tiny_sample: dict[str, Path]) -> None:
    # draw a single pixel and ensure nearest keeps 0/255 after resize
    from PIL import Image as PILImage

    m = np.zeros((64, 80), dtype=np.uint8)
    m[20, 30] = 255
    PILImage.fromarray(m).save(tiny_sample["mask_path"])
    out = read_rgb_depth_mask(
        tiny_sample["image_path"],
        tiny_sample["depth_path"],
        tiny_sample["mask_path"],
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
        use_upstream_resize_convention=True,
    )
    uniq = set(np.unique(out["mask_hw"]).tolist())
    assert uniq.issubset({0, 255})
