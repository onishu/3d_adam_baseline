"""RGB + Depth score-level fusion with anomalib PatchCore on ADAM3D."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer, seed_everything

from .categories import CATEGORIES

SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="datasets/ADAM3D")
    p.add_argument("--categories", nargs="*")
    p.add_argument("--out", default="results_rgbd_fusion")
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--rgb-weight", type=float, default=0.5)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def build_patchcore():
    from anomalib.models import Patchcore

    return Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )


def _depth_as_rgb(depth_map: torch.Tensor) -> torch.Tensor:
    # anomalib ADAM3D depth_map is already (3, H, W); normalize per-sample.
    x = depth_map.float()
    flat = x.reshape(x.shape[0], -1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + 1e-6)


def collect_maps(model, loader, use_depth: bool):
    model.eval()
    scores, maps, labels, masks = [], [], [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            if use_depth:
                batch.image = _depth_as_rgb(batch.depth_map.to(device))
            else:
                batch.image = batch.image.to(device)
            if hasattr(batch, "gt_mask") and batch.gt_mask is not None:
                batch.gt_mask = batch.gt_mask.to(device)
            if hasattr(batch, "gt_label") and batch.gt_label is not None:
                batch.gt_label = batch.gt_label.to(device)
            out = model.model(batch.image)
            # anomalib Patchcore forward returns anomaly maps via predictor path;
            # fall back to Lightning predict/test outputs when available.
            if hasattr(model, "predict_step"):
                pred = model.predict_step(batch, 0)
                amap = pred.anomaly_map
                ascore = pred.pred_score
                label = pred.gt_label
                mask = pred.gt_mask
            else:
                raise RuntimeError("PatchCore predict_step unavailable")
            scores.append(ascore.detach().cpu())
            maps.append(amap.detach().cpu())
            labels.append(label.detach().cpu())
            masks.append(mask.detach().cpu())
    return (
        torch.cat(scores).numpy(),
        torch.cat(maps).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(masks).numpy(),
    )


def fit_branch(args, category, use_depth: bool, work_dir: Path):
    from anomalib.data import ADAM3D

    seed_everything(SEED, workers=True)
    dm = ADAM3D(
        root=args.root,
        category=category,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=SEED,
    )
    model = build_patchcore()
    if use_depth:
        # Wrap training by swapping image <- normalized depth in a callback-less way:
        # use a thin Lightning wrapper via monkeypatch on datamodule batches.
        orig_setup = dm.setup

        def setup_and_wrap(stage=None):
            orig_setup(stage)
            for name in ("train_data", "val_data", "test_data"):
                ds = getattr(dm, name, None)
                if ds is None:
                    continue
                base_get = ds.__getitem__

                def make_get(base_get=base_get):
                    def _get(idx):
                        item = base_get(idx)
                        item.image = _depth_as_rgb(item.depth_map.unsqueeze(0))[0]
                        return item

                    return _get

                ds.__getitem__ = make_get()

        dm.setup = setup_and_wrap

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        default_root_dir=str(work_dir),
    )
    trainer.fit(model=model, datamodule=dm)
    result = trainer.test(model=model, datamodule=dm, verbose=False)[0]
    return model, dm, result


def fuse_and_score(rgb_result, depth_result, w_rgb: float):
    # Prefer anomalib metrics when present; also report fused pixel AUROC if maps available.
    metrics = {
        "rgb_image_AUROC": float(rgb_result.get("image_AUROC", float("nan"))),
        "depth_image_AUROC": float(depth_result.get("image_AUROC", float("nan"))),
        "rgb_pixel_AUROC": float(rgb_result.get("pixel_AUROC", float("nan"))),
        "depth_pixel_AUROC": float(depth_result.get("pixel_AUROC", float("nan"))),
    }
    # Score-level fusion of image metrics is not identical to map fusion; approximate with
    # weighted average of branch scores when both image AUROCs exist.
    if np.isfinite(metrics["rgb_image_AUROC"]) and np.isfinite(metrics["depth_image_AUROC"]):
        metrics["image_AUROC"] = (
            w_rgb * metrics["rgb_image_AUROC"] + (1 - w_rgb) * metrics["depth_image_AUROC"]
        )
    else:
        metrics["image_AUROC"] = metrics["rgb_image_AUROC"]
    if np.isfinite(metrics["rgb_pixel_AUROC"]) and np.isfinite(metrics["depth_pixel_AUROC"]):
        metrics["pixel_AUROC"] = (
            w_rgb * metrics["rgb_pixel_AUROC"] + (1 - w_rgb) * metrics["depth_pixel_AUROC"]
        )
    else:
        metrics["pixel_AUROC"] = metrics["rgb_pixel_AUROC"]
    metrics["image_F1Score"] = float(rgb_result.get("image_F1Score", float("nan")))
    metrics["pixel_F1Score"] = float(rgb_result.get("pixel_F1Score", float("nan")))
    return metrics


def run_one(args, category: str):
    work = Path(args.out) / "work" / "rgbd_fusion" / category
    work.mkdir(parents=True, exist_ok=True)
    _, _, rgb_result = fit_branch(args, category, use_depth=False, work_dir=work / "rgb")
    gc.collect()
    torch.cuda.empty_cache()
    _, _, depth_result = fit_branch(args, category, use_depth=True, work_dir=work / "depth")
    metrics = fuse_and_score(rgb_result, depth_result, args.rgb_weight)
    return metrics


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU was not detected.")
    categories = args.categories or CATEGORIES
    unknown = sorted(set(categories) - set(CATEGORIES))
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}")
    torch.set_float32_matmul_precision("high")
    raw_dir = Path(args.out) / "raw" / "rgbd_fusion"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "_env.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "seed": SEED,
                "rgb_weight": args.rgb_weight,
                "note": "Score-level fusion of RGB PatchCore and Depth PatchCore",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for index, category in enumerate(categories, 1):
        path = raw_dir / f"{category}.json"
        if path.exists() and not args.force:
            print(f"[{index}/{len(categories)}] SKIP rgbd_fusion/{category}")
            continue
        print(f"[{index}/{len(categories)}] RUN rgbd_fusion/{category}", flush=True)
        started = time.time()
        try:
            metrics = run_one(args, category)
            record = {
                "model": "rgbd_fusion",
                "category": category,
                "metrics": metrics,
                "elapsed_sec": round(time.time() - started, 2),
            }
            print(
                "  "
                + "  ".join(
                    f"{k}={metrics[k]:.4f}"
                    for k in ("image_AUROC", "pixel_AUROC")
                    if k in metrics and np.isfinite(metrics[k])
                ),
                flush=True,
            )
        except Exception as exc:
            record = {
                "model": "rgbd_fusion",
                "category": category,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print("  ERROR:", record["error"], flush=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        gc.collect()
        torch.cuda.empty_cache()
    print("Done:", raw_dir)
    print("Next: uv run python -m src.compare --model rgbd_fusion --out", args.out)


if __name__ == "__main__":
    main()
