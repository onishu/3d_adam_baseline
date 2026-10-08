"""Adapter for PaulMcHard/3D-ADAMBench TransFusion on anomalib ADAM3D.

Root-cause note
---------------
Upstream ``get_plane_mask`` estimates a background plane from the four image
corners (MVTec3D assumption). On ADAM3D anomalib-subset XYZ TIFFs, corners are
all Z==0 (invalid), which causes divide-by-zero and an all-zero foreground
mask. Synthetic anomalies are then multiplied by this mask and become empty.

This adapter replaces plane estimation with a valid-point fit and validates
DTD / shapes before training.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any  # noqa: F401 — used throughout

import numpy as np

from src.common.config import REPO_ROOT
from src.common.finite_check import assert_finite
from src.common.hashing import sha256_file

logger = logging.getLogger(__name__)

TF_ROOT = REPO_ROOT / "third_party" / "3D-ADAMBench" / "TransFusion_3D-ADAM"
UPSTREAM_ADAM_BENCH = {
    "url": "https://github.com/PaulMcHard/3D-ADAMBench",
    "commit": "58c7e108a4b1de5789486fd2917245a1907234bf",
    "license": "MIT (TransFusion_3D-ADAM/LICENSE; Copyright 2024 Matic Fučka)",
}
UPSTREAM_ECCV = {
    "url": "https://github.com/maticfuc/ECCV_TransFusion",
    "commit": "a419dd93f26fa3c140fadf6d35fdf1a79074a515",
    "license": "MIT",
}

# Canonical spatial layout after adapter normalization:
# image tensor: (C, H, W) with H=height=256, W=width=320
# NOTE: upstream Dataset uses torchvision.Resize((width, height)) which treats
# the pair as (h, w), producing tensors shaped (C, 320, 256). We document and
# test that upstream convention explicitly; adapter can optionally canonicalize.
UPSTREAM_SPATIAL_H = 320  # first spatial dim in upstream tensors
UPSTREAM_SPATIAL_W = 256  # second spatial dim in upstream tensors
TARGET_WIDTH = 320
TARGET_HEIGHT = 256


@dataclass
class TransfusionPaths:
    tf_root: Path = TF_ROOT
    dtd_images: Path = REPO_ROOT / "datasets" / "dtd" / "images"
    fg_mask_root: Path = REPO_ROOT / "fg_masks"


def ensure_tf_on_path(*, chdir: bool = True) -> Path:
    if not TF_ROOT.is_dir():
        raise FileNotFoundError(f"Missing TransFusion sources: {TF_ROOT}")
    # Prefer TransFusion package modules over similarly named third_party modules.
    cleaned = [p for p in sys.path if "AnomalyDetection-M3DM" not in p]
    if str(TF_ROOT) not in cleaned:
        cleaned = [str(TF_ROOT), *cleaned]
    elif cleaned[0] != str(TF_ROOT):
        cleaned = [str(TF_ROOT), *[p for p in cleaned if p != str(TF_ROOT)]]
    sys.path[:] = cleaned
    if chdir:
        os.chdir(TF_ROOT)
    return TF_ROOT


def validate_dtd(dtd_images: Path, min_images: int = 100) -> dict[str, Any]:
    """Validate DTD texture root used for synthetic RGB anomalies."""
    info: dict[str, Any] = {
        "dtd_root": str(dtd_images).replace("\\", "/"),
        "exists": dtd_images.is_dir(),
        "n_images": 0,
        "ok": False,
    }
    if not dtd_images.is_dir():
        return info
    # Deduplicate: on Windows *.jpg and *.JPG match the same files.
    jpgs = sorted({p.resolve() for p in dtd_images.rglob("*.jpg")})
    info["n_images"] = len(jpgs)
    info["ok"] = len(jpgs) >= min_images
    if info["ok"] and jpgs:
        # Hash a stable manifest of relative paths (not file contents) for audit.
        rels = sorted(str(p.relative_to(dtd_images.resolve())).replace("\\", "/") for p in jpgs)
        from src.common.hashing import sha256_text

        info["path_manifest_sha256"] = sha256_text("\n".join(rels))
    return info


FOREGROUND_STRATEGIES = (
    "upstream_corner_fit",
    "adam3d_valid_point_fit",  # legacy alias → valid_point_plane_fit_with_depth_fallback
    "valid_point_plane_fit",
    "valid_point_plane_fit_with_depth_fallback",
    "valid_depth_mask",
    "provided_foreground_mask",
)

FOREGROUND_ALIASES = {
    "valid-depth-mask": "valid_depth_mask",
    "valid_depth": "valid_depth_mask",
    "adam3d": "valid_point_plane_fit_with_depth_fallback",
    "adam3d_valid_point_fit": "valid_point_plane_fit_with_depth_fallback",
    "valid-point-plane-fit": "valid_point_plane_fit",
    "valid-point-plane-fit-with-depth-fallback": "valid_point_plane_fit_with_depth_fallback",
    "upstream": "upstream_corner_fit",
    "provided": "provided_foreground_mask",
}


def normalize_foreground_strategy(name: str) -> str:
    key = str(name).strip()
    key = FOREGROUND_ALIASES.get(key, key)
    if key not in FOREGROUND_STRATEGIES:
        raise ValueError(f"Unknown foreground strategy: {name}")
    return key


def _valid_depth_mask(depth_image: np.ndarray) -> np.ndarray:
    z = depth_image[:, :, 2]
    return (z != 0).astype(np.float32)[:, :, None]


def get_plane_mask_upstream_corner(depth_image: np.ndarray, thr: float = 0.005) -> np.ndarray:
    """Original upstream corner-plane fit (may collapse on ADAM3D Z==0 corners)."""
    ensure_tf_on_path(chdir=False)
    from utils import geo_utils

    # Call the *original* implementation if we still hold a reference; else inline.
    orig = getattr(geo_utils, "_original_get_plane_mask", None)
    if orig is not None:
        return orig(depth_image)
    # Inline replica of upstream geo_utils.get_plane_mask
    from utils.geo_utils import get_distance_to_plane, get_plane_from_points

    h, w, c = depth_image.shape
    points = np.reshape(depth_image, (h * w, c))
    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = np.sum(depth_image[:3, :3, :], axis=(0, 1)) / np.sum(
            depth_image[:3, :3, 2] != 0
        )
        p2 = np.sum(depth_image[:3, -3:, :], axis=(0, 1)) / np.sum(
            depth_image[:3, -3:, 2] != 0
        )
        p3 = np.sum(depth_image[-3:, :3, :], axis=(0, 1)) / np.sum(
            depth_image[-3:, :3, 2] != 0
        )
    plane = get_plane_from_points(p1, p2, p3)
    point_distance = get_distance_to_plane(points, np.array(plane))
    points_mask = np.where(
        point_distance > thr,
        np.ones_like(point_distance),
        np.zeros_like(point_distance),
    )
    return np.reshape(points_mask, (h, w, 1)).astype(np.float32)


def diagnose_plane_fit(
    depth_image: np.ndarray, thr: float = 0.005
) -> dict[str, Any]:
    """Return detailed plane-fit diagnostics for one XYZ image (no fallback applied)."""
    ensure_tf_on_path(chdir=False)
    from utils.geo_utils import get_distance_to_plane, get_plane_from_points

    if depth_image.ndim != 3 or depth_image.shape[-1] < 3:
        raise ValueError(f"Expected HxWx3 XYZ, got {depth_image.shape}")
    h, w, _ = depth_image.shape
    z = depth_image[:, :, 2]
    valid = z != 0
    pts = depth_image[valid]
    n_valid = int(valid.sum())
    meta: dict[str, Any] = {
        "valid_point_count": n_valid,
        "sampled_point_count": 0,
        "coordinate_min": pts.min(axis=0).tolist() if n_valid else None,
        "coordinate_max": pts.max(axis=0).tolist() if n_valid else None,
        "coordinate_scale": float(np.linalg.norm(pts.std(axis=0))) if n_valid else None,
        "matrix_rank": None,
        "singular_values": None,
        "estimated_normal": None,
        "plane_offset": None,
        "residual_min": None,
        "residual_mean": None,
        "residual_max": None,
        "inlier_threshold": thr,
        "inlier_count": None,
        "foreground_count": None,
        "failure_reason": None,
        "fallback_used": False,
    }
    if n_valid < 3:
        meta["failure_reason"] = "insufficient_valid_points"
        meta["fallback_used"] = True
        return meta

    ys, xs = np.where(valid)
    i1 = int(np.argmin(ys))
    i2 = int(np.argmax(xs))
    i3 = int(np.argmax(ys))
    if len({i1, i2, i3}) < 3:
        idx = np.linspace(0, len(ys) - 1, 3, dtype=int)
        i1, i2, i3 = (int(i) for i in idx)
    sample = np.stack(
        [
            depth_image[ys[i1], xs[i1]],
            depth_image[ys[i2], xs[i2]],
            depth_image[ys[i3], xs[i3]],
        ],
        axis=0,
    )
    meta["sampled_point_count"] = 3
    # Centered design matrix rank / singular values for the three points.
    centered = sample - sample.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        meta["singular_values"] = s.tolist()
        meta["matrix_rank"] = int(np.linalg.matrix_rank(centered))
    except Exception as exc:  # noqa: BLE001
        meta["failure_reason"] = f"svd_failed:{exc}"
        meta["fallback_used"] = True
        return meta

    plane = get_plane_from_points(sample[0], sample[1], sample[2])
    normal = np.asarray(plane[:3], dtype=np.float64)
    meta["estimated_normal"] = normal.tolist()
    meta["plane_offset"] = float(plane[3]) if len(plane) > 3 else None
    if (not np.isfinite(plane).all()) or float(np.sum(normal**2)) < 1e-12:
        meta["failure_reason"] = "degenerate_plane_coefficients"
        meta["fallback_used"] = True
        return meta

    dist = get_distance_to_plane(depth_image.reshape(-1, 3), np.array(plane))
    dist_valid = dist[valid.reshape(-1)]
    if not np.isfinite(dist).all():
        meta["failure_reason"] = "nonfinite_residuals"
        meta["fallback_used"] = True
        return meta
    meta["residual_min"] = float(np.min(dist_valid)) if dist_valid.size else None
    meta["residual_mean"] = float(np.mean(dist_valid)) if dist_valid.size else None
    meta["residual_max"] = float(np.max(dist_valid)) if dist_valid.size else None
    mask = (dist > thr).reshape(h, w) & valid
    # "Inliers" to the background plane = low residual among valid points.
    inliers = (dist.reshape(h, w) <= thr) & valid
    meta["inlier_count"] = int(inliers.sum())
    meta["foreground_count"] = int(mask.sum())
    rate = float(mask.mean())
    if rate < 0.01:
        meta["failure_reason"] = f"foreground_ratio_too_low:{rate:.6f}"
        meta["fallback_used"] = True
    elif rate > 0.99:
        meta["failure_reason"] = f"foreground_ratio_too_high:{rate:.6f}"
        meta["fallback_used"] = True
    return meta


def get_plane_mask_adam3d(
    depth_image: np.ndarray,
    thr: float = 0.005,
    *,
    return_meta: bool = False,
    allow_fallback: bool = True,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Foreground via 3-point valid-XYZ plane fit (legacy name kept).

    On organized ADAM3D XYZ, the object often lies near a plane; residuals can
    be tiny so the ``dist > thr`` foreground collapses. When ``allow_fallback``
    the valid-depth mask is returned (strategy name:
    ``valid_point_plane_fit_with_depth_fallback``).
    """
    ensure_tf_on_path(chdir=False)
    from utils.geo_utils import get_distance_to_plane, get_plane_from_points

    diag = diagnose_plane_fit(depth_image, thr=thr)
    meta: dict[str, Any] = {
        "degenerate": bool(diag.get("failure_reason")),
        "fallback_used": False,
        "failure_reason": diag.get("failure_reason"),
        "diagnostics": diag,
    }
    h, w, _ = depth_image.shape
    valid = depth_image[:, :, 2] != 0
    if diag.get("failure_reason") in {
        "insufficient_valid_points",
        "svd_failed",
        "degenerate_plane_coefficients",
        "nonfinite_residuals",
    } or (
        isinstance(diag.get("failure_reason"), str)
        and diag["failure_reason"].startswith("svd_failed")
    ):
        if not allow_fallback:
            raise RuntimeError(diag.get("failure_reason"))
        meta["fallback_used"] = True
        out = valid.astype(np.float32)[:, :, None]
        return (out, meta) if return_meta else out

    ys, xs = np.where(valid)
    i1 = int(np.argmin(ys))
    i2 = int(np.argmax(xs))
    i3 = int(np.argmax(ys))
    if len({i1, i2, i3}) < 3:
        idx = np.linspace(0, len(ys) - 1, 3, dtype=int)
        i1, i2, i3 = (int(i) for i in idx)
    p1 = depth_image[ys[i1], xs[i1]]
    p2 = depth_image[ys[i2], xs[i2]]
    p3 = depth_image[ys[i3], xs[i3]]
    plane = get_plane_from_points(p1, p2, p3)
    dist = get_distance_to_plane(depth_image.reshape(-1, 3), np.array(plane))
    mask = (dist > thr).reshape(h, w) & valid
    rate = float(mask.mean())
    if rate < 0.01 or rate > 0.99:
        if allow_fallback:
            logger.warning(
                "plane mask rate=%.4f outside [0.01,0.99]; using valid-depth mask (%s)",
                rate,
                diag.get("failure_reason"),
            )
            meta.update(degenerate=True, fallback_used=True)
            out = valid.astype(np.float32)[:, :, None]
            return (out, meta) if return_meta else out
        out = mask.astype(np.float32)[:, :, None]
        return (out, meta) if return_meta else out
    out = mask.astype(np.float32)[:, :, None]
    return (out, meta) if return_meta else out


def make_plane_mask_fn(
    strategy: str = "valid_point_plane_fit_with_depth_fallback", thr: float = 0.005
):
    """Return a get_plane_mask-compatible callable for ``strategy``."""
    strategy = normalize_foreground_strategy(strategy)

    if strategy == "upstream_corner_fit":
        return lambda depth: get_plane_mask_upstream_corner(depth, thr=thr)
    if strategy == "valid_depth_mask":
        return lambda depth: _valid_depth_mask(depth)
    if strategy == "provided_foreground_mask":
        return lambda depth: _valid_depth_mask(depth)
    if strategy == "valid_point_plane_fit":
        return lambda depth: get_plane_mask_adam3d(
            depth, thr=thr, allow_fallback=False
        )
    # adam3d_valid_point_fit / valid_point_plane_fit_with_depth_fallback
    return lambda depth: get_plane_mask_adam3d(depth, thr=thr, allow_fallback=True)


def install_plane_mask_patch(
    strategy: str = "valid_point_plane_fit_with_depth_fallback",
) -> None:
    """Monkeypatch upstream geo_utils.get_plane_mask with selected strategy."""
    strategy = normalize_foreground_strategy(strategy)
    ensure_tf_on_path()
    import utils.geo_utils as geo_utils

    if not hasattr(geo_utils, "_original_get_plane_mask"):
        geo_utils._original_get_plane_mask = geo_utils.get_plane_mask  # type: ignore[attr-defined]
    geo_utils.get_plane_mask = make_plane_mask_fn(strategy)  # type: ignore[attr-defined]
    logger.info("Installed get_plane_mask patch strategy=%s", strategy)


def install_amp_patch(enabled: bool) -> None:
    """Force TransFusion.loss autocast on/off without editing upstream file body permanently."""
    ensure_tf_on_path()
    import torch

    # Avoid wrapping our own wrapper when install_amp_patch is called repeatedly.
    real = getattr(torch.cuda.amp, "_original_autocast", None)
    if real is None:
        real = torch.cuda.amp.autocast
        torch.cuda.amp._original_autocast = real  # type: ignore[attr-defined]

    def _autocast(*args, **kwargs):  # noqa: ANN002, ANN003
        # Upstream calls autocast(True) positionally; always override enabled.
        kwargs["enabled"] = bool(enabled)
        if args:
            # drop positional enabled if present
            args = args[1:] if len(args) >= 1 else ()
        return real(*args, **kwargs)

    torch.cuda.amp.autocast = _autocast  # type: ignore[assignment]
    logger.info("Installed AMP patch enabled=%s", enabled)


def _abs_under_repo(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def build_namespace(
    *,
    data_path: Path,
    category: str,
    mode: str = "rgbd",
    img_width: int = TARGET_WIDTH,
    img_height: int = TARGET_HEIGHT,
    dtd_path: Path | None = None,
    fg_mask_path: Path | None = None,
    seed: int = 3,
    no_fg_masks: bool = True,
    diffusion_num_steps: int = 75,
) -> Any:
    import argparse

    paths = TransfusionPaths()
    return argparse.Namespace(
        data_path=str(_abs_under_repo(data_path)),
        dataset="adam3d",
        category=category,
        mode=mode,
        img_width=img_width,
        img_height=img_height,
        dtd_path=str(_abs_under_repo(dtd_path or paths.dtd_images)),
        no_fg_masks=no_fg_masks,
        fg_mask_path=str(_abs_under_repo(fg_mask_path or paths.fg_mask_root)),
        seed=seed,
        diffusion_num_steps=diffusion_num_steps,
        unet_channel_num=64,
        unet_ch_mults=[1, 2, 2, 2],
        unet_attn=[False, False, True, True],
        unet_group_conv=[True, True, True, True],
        unet_n_blocks=2,
        lr=1e-5,
        step_size=800,
        gamma=0.1,
    )


def load_train_dataset(
    args: Any, *, foreground_strategy: str = "valid_point_plane_fit_with_depth_fallback"
):
    ensure_tf_on_path()
    install_plane_mask_patch(strategy=foreground_strategy)
    from utils.ParsingUtils import get_dataset

    return get_dataset(args, True)


def load_test_dataset(
    args: Any, *, foreground_strategy: str = "valid_point_plane_fit_with_depth_fallback"
):
    ensure_tf_on_path()
    install_plane_mask_patch(strategy=foreground_strategy)
    from utils.ParsingUtils import get_dataset

    return get_dataset(args, False)


def load_model(args: Any):
    ensure_tf_on_path()
    from utils.ParsingUtils import create_model

    return create_model(args)


def sample_raw_paths(dataset_root: Path, category: str, split: str = "train") -> dict[str, Path]:
    """Pick one RGB/XYZ(/mask) triple for adapter unit tests."""
    if split == "train":
        rgb_dir = dataset_root / category / "train" / "good" / "rgb"
        rgb = sorted(rgb_dir.glob("*.png"))[0]
        xyz = dataset_root / category / "train" / "good" / "xyz" / f"{rgb.stem}.tiff"
        return {"image_path": rgb, "depth_path": xyz, "mask_path": Path()}
    # first test defect
    test_root = dataset_root / category / "test"
    for label_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        rgb_dir = label_dir / "rgb"
        if not rgb_dir.is_dir():
            continue
        rgb = sorted(rgb_dir.glob("*.png"))[0]
        xyz = label_dir / "xyz" / f"{rgb.stem}.tiff"
        mask = label_dir / "ground_truth" / f"{rgb.stem}.png"
        return {"image_path": rgb, "depth_path": xyz, "mask_path": mask}
    raise FileNotFoundError(f"No test RGB found under {test_root}")


def read_rgb_depth_mask(
    image_path: Path,
    depth_path: Path,
    mask_path: Path | None = None,
    *,
    width: int = TARGET_WIDTH,
    height: int = TARGET_HEIGHT,
    use_upstream_resize_convention: bool = True,
) -> dict[str, Any]:
    """Load one sample with explicit resize conventions.

    When ``use_upstream_resize_convention`` is True, matches upstream Dataset:
    torchvision.Resize((width, height)) => output spatial (width, height) as (H, W),
    cv2.resize(dsize=(width, height)) => array (height, width).
    """
    from PIL import Image
    import cv2
    import tifffile
    import torch
    import torchvision

    rgb = Image.open(image_path).convert("RGB")
    if use_upstream_resize_convention:
        # Upstream bug/convention: passes (width, height) into Resize which expects (h, w).
        rgb_t = torchvision.transforms.Resize((width, height))(rgb)
    else:
        rgb_t = torchvision.transforms.Resize((height, width))(rgb)
    rgb_arr = np.array(rgb_t)
    rgb_chw = torch.FloatTensor(((rgb_arr / 255.0 - 0.5) * 2).transpose(2, 0, 1))

    xyz = tifffile.imread(str(depth_path)).astype(np.float32)
    if use_upstream_resize_convention:
        xyz_r = cv2.resize(xyz, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        xyz_r = cv2.resize(xyz, (width, height), interpolation=cv2.INTER_NEAREST)
        # canonical: dsize is always (W, H) for cv2
    if xyz_r.ndim == 2:
        raise ValueError("Expected 3-channel XYZ TIFF")
    z = xyz_r[:, :, 2]
    assert_finite("raw_resized_xyz", xyz_r, enabled=True)

    mask_arr = None
    if mask_path is not None and Path(mask_path).is_file():
        mask_img = Image.open(mask_path).convert("L")
        if use_upstream_resize_convention:
            mask_img = torchvision.transforms.Resize(
                (width, height), interpolation=torchvision.transforms.InterpolationMode.NEAREST
            )(mask_img)
        else:
            mask_img = torchvision.transforms.Resize(
                (height, width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
            )(mask_img)
        mask_arr = np.array(mask_img)

    return {
        "rgb_chw": rgb_chw,
        "xyz_hwc": xyz_r,
        "z_hw": z,
        "mask_hw": mask_arr,
        "rgb_spatial": tuple(rgb_chw.shape[-2:]),
        "xyz_spatial": tuple(xyz_r.shape[:2]),
    }


def expect_upstream_tensor_spatial(img_width: int = TARGET_WIDTH, img_height: int = TARGET_HEIGHT) -> tuple[int, int]:
    """Return (H, W) produced by upstream Dataset for given CLI width/height."""
    # torchvision.Resize((img_width, img_height)) => H=img_width, W=img_height
    return (img_width, img_height)
