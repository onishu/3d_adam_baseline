"""Adapter-backed TransFusion full-train / pilot loop."""

from __future__ import annotations

import csv
import json
import logging
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.benchmarks.table5.checkpoint import (
    build_checkpoint,
    build_inference_checkpoint,
    load_training_checkpoint,
    prune_periodic_checkpoints,
    save_inference_checkpoint,
    save_training_checkpoint,
)
from src.common.config import REPO_ROOT
from src.common.finite_check import assert_finite, assert_finite_many
from src.common.metrics import (
    AUPROConfig,
    aupro,
    author_aupro,
    best_f1_threshold,
    image_auroc,
    pixel_auroc,
)
from src.common.result_schema import ExperimentResult, ResultStatus, atomic_write_json, utc_now_iso
from src.data.adapters import transfusion as tf_adapter

logger = logging.getLogger("benchmarks.table5.transfusion.train")

CHECKPOINT_SELECTION_METRICS = (
    "validation_image_auroc",
    "validation_pixel_auroc",
    "validation_author_aupro",
    "validation_total_loss",
    "last",
)

# Author Experiment.py overwrites the single model.pkl at each eval epoch;
# there is no metric-based "best" selection. Default follows that policy.
DEFAULT_CHECKPOINT_SELECTION = "last"
DEFAULT_CHECKPOINT_SELECTION_SOURCE = "author_code"


def git_head_sha(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return None


def _grad_norm(model: torch.nn.Module) -> float:
    grads = [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.stack(grads).norm().cpu())


def _param_norm(model: torch.nn.Module) -> float:
    ps = [p.detach().float().norm() for p in model.parameters()]
    if not ps:
        return 0.0
    return float(torch.stack(ps).norm().cpu())


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _validation_overlap(manifest: Path, category: str) -> dict[str, Any]:
    import csv as _csv

    val: set[str] = set()
    test: set[str] = set()
    with Path(manifest).open(encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if row["category"] != category:
                continue
            key = row.get("image_path") or row.get("rgb_path") or row.get("stem") or json.dumps(row)
            if row["split"] == "validation":
                val.add(key)
            elif row["split"] == "test":
                test.add(key)
    inter = val & test
    return {
        "validation_test_overlap_count": len(inter),
        "validation_test_identical": val == test and len(val) > 0,
        "validation_count": len(val),
        "test_count": len(test),
    }


def preflight_full_train(
    *,
    args: Any,
    runtime: dict[str, Any],
    category: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Raise SystemExit if full-train preconditions fail."""
    dtd = runtime["dtd"]
    if int(dtd.get("n_images", 0)) < 100 or not dtd.get("ok"):
        raise SystemExit(f"DTD precondition failed: {dtd}")
    if not dtd.get("path_manifest_sha256"):
        raise SystemExit("DTD path_manifest_sha256 missing")

    man = Path(runtime["manifest"])
    if not man.is_file():
        raise SystemExit(f"Missing manifest: {man}")
    val_json = man.with_suffix(".validation.json")
    if val_json.is_file():
        report = json.loads(val_json.read_text(encoding="utf-8"))
        if report.get("errors"):
            raise SystemExit(f"Manifest validation has errors: {val_json}")

    split = runtime.get("split_info") or {}
    counts = split.get("counts") or {}
    for key in ("train", "validation", "test"):
        if int(counts.get(key, 0)) <= 0:
            raise SystemExit(f"Empty {key} set for {category}: {counts}")

    raw_done = out_dir / "raw" / "transfusion" / f"{category}.json"
    if raw_done.is_file() and not getattr(args, "force", False) and not getattr(args, "resume", False):
        raise SystemExit(
            f"Refusing to overwrite completed result {raw_done}; pass --force or --resume"
        )

    return {"ok": True, "overlap": _validation_overlap(man, category)}


def _seed_eval(seed: int) -> None:
    """Seed RNGs before diffusion-based evaluation for checkpoint-stable metrics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_paths(dataset) -> list[str]:
    inner = dataset
    while hasattr(inner, "dataset"):
        inner = inner.dataset
    files = getattr(inner, "_files", None)
    if not files:
        return [str(i) for i in range(len(dataset))]
    return [str(Path(p).as_posix()) for p in files]


def _predict_maps(
    net: torch.nn.Module,
    dataset,
    *,
    max_steps: int | None,
    kernel_size: int,
    weight: float,
    fail: bool,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[str]]:
    """Run author EvaluationUtils path for anomaly maps (eval / no_grad)."""
    from scipy import signal

    tf_adapter.ensure_tf_on_path()
    from utils import EvaluationUtils

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    kern = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    maps: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    file_paths = _dataset_paths(dataset)
    if seed is not None:
        _seed_eval(seed)
    was_training = net.training
    net.eval()
    with torch.inference_mode():
        for i, data in enumerate(loader):
            if max_steps is not None and i >= max_steps:
                break
            # Per-sample seed so sample order / count changes do not scramble later maps.
            if seed is not None:
                _seed_eval(seed + i)
            reconstructed, mask_disc, mask_recon = EvaluationUtils.calculate_transfussion_results(
                net, dataset, data["index"]
            )
            mask_disc = signal.convolve2d(mask_disc.squeeze(), kern, mode="same")
            mask_recon = signal.convolve2d(mask_recon.squeeze(), kern, mode="same")
            mask_final = weight * mask_disc + (1.0 - weight) * mask_recon
            gt = (data["mask"].numpy().squeeze() > 0.5).astype(np.int32)
            assert_finite("anomaly_map", torch.as_tensor(mask_final), enabled=fail)
            maps.append(np.asarray(mask_final, dtype=np.float64))
            gts.append(gt)
            if "is_anomaly" in data:
                lab = data["is_anomaly"]
                labels.append(int(lab.item() if torch.is_tensor(lab) else lab))
            else:
                labels.append(int(gt.max() > 0))
            idx = int(data["index"].cpu().numpy().reshape(-1)[0])
            paths.append(file_paths[idx] if idx < len(file_paths) else str(idx))
    if was_training:
        net.train()
    return maps, gts, labels, paths


def _filter_empty_gt(
    maps: list[np.ndarray],
    gts: list[np.ndarray],
    labels: list[int],
    policy: str,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    empty_idx = [i for i, g in enumerate(gts) if int(labels[i]) == 1 and int(np.asarray(g).sum()) == 0]
    meta = {
        "total_test_images": len(gts),
        "empty_gt_count": len(empty_idx),
        "pixel_metric_policy": policy,
    }
    if policy == "include_empty_anomalous_masks":
        meta["included_empty_gt_count"] = len(empty_idx)
        meta["excluded_empty_gt_count"] = 0
        return maps, gts, meta
    keep = [i for i in range(len(gts)) if i not in set(empty_idx)]
    meta["included_empty_gt_count"] = 0
    meta["excluded_empty_gt_count"] = len(empty_idx)
    return [maps[i] for i in keep], [gts[i] for i in keep], meta


def _score_stats(maps: list[np.ndarray], labels: list[int]) -> dict[str, Any]:
    img_scores = np.array([float(m.max()) for m in maps], dtype=np.float64)
    labels_a = np.array(labels, dtype=np.int32)
    normals = img_scores[labels_a == 0]
    anoms = img_scores[labels_a == 1]
    stacked = np.concatenate([m.reshape(-1) for m in maps]) if maps else np.array([0.0])
    thr, f1 = (None, None)
    if len(np.unique(labels_a)) >= 2:
        thr, f1 = best_f1_threshold(labels_a, img_scores)
    return {
        "image_score_mean_normal": float(normals.mean()) if normals.size else None,
        "image_score_mean_anomalous": float(anoms.mean()) if anoms.size else None,
        "image_score_std_normal": float(normals.std()) if normals.size else None,
        "image_score_std_anomalous": float(anoms.std()) if anoms.size else None,
        "anomaly_map_mean": float(stacked.mean()),
        "anomaly_map_std": float(stacked.std()),
        "anomaly_map_min": float(stacked.min()),
        "anomaly_map_max": float(stacked.max()),
        "anomaly_map_unique_approx": int(min(len(np.unique(stacked[:100000])), 100000)),
        "selected_threshold": thr,
        "selected_threshold_f1": f1,
        "image_scores": img_scores.tolist(),
    }


def compute_eval_metrics(
    maps: list[np.ndarray],
    gts: list[np.ndarray],
    labels: list[int],
    *,
    empty_policy: str,
    aupro_fpr_limit: float = 0.3,
    threshold_provenance: str = "validation",
) -> dict[str, Any]:
    stats = _score_stats(maps, labels)
    img_scores = np.array(stats["image_scores"], dtype=np.float64)
    img_labels = np.array(labels, dtype=np.int32)
    out: dict[str, Any] = {k: v for k, v in stats.items() if k != "image_scores"}
    out["threshold_provenance"] = threshold_provenance
    try:
        out["image_AUROC"] = image_auroc(img_labels, img_scores)
    except ValueError as exc:
        out["image_AUROC"] = None
        out["image_AUROC_error"] = str(exc)

    maps_f, gts_f, pix_meta = _filter_empty_gt(maps, gts, labels, empty_policy)
    out.update(pix_meta)
    try:
        yt = np.concatenate([g.reshape(-1) for g in gts_f]) if gts_f else np.array([])
        ys = np.concatenate([m.reshape(-1) for m in maps_f]) if maps_f else np.array([])
        out["pixel_AUROC"] = pixel_auroc(yt, ys) if yt.size else None
    except ValueError as exc:
        out["pixel_AUROC"] = None
        out["pixel_AUROC_error"] = str(exc)

    try:
        out["common_aupro"] = aupro(
            gts_f, maps_f, AUPROConfig(fpr_limit=aupro_fpr_limit)
        )
    except Exception as exc:  # noqa: BLE001
        out["common_aupro"] = None
        out["common_aupro_error"] = str(exc)
    try:
        out["author_aupro"] = author_aupro(maps_f, gts_f)
    except Exception as exc:  # noqa: BLE001
        out["author_aupro"] = None
        out["author_aupro_error"] = str(exc)
    out["AUPRO"] = out.get("author_aupro")
    return out


def _selection_value(
    metric_name: str,
    metrics: dict[str, Any],
    train_loss: float | None,
) -> float | None:
    if metric_name == "last":
        return None
    if metric_name == "validation_image_auroc":
        return metrics.get("image_AUROC")
    if metric_name == "validation_pixel_auroc":
        return metrics.get("pixel_AUROC")
    if metric_name == "validation_author_aupro":
        return metrics.get("author_aupro")
    if metric_name == "validation_total_loss":
        return train_loss
    raise ValueError(metric_name)


def _is_better(metric_name: str, new: float, old: float) -> bool:
    if metric_name == "validation_total_loss":
        return new < old
    return new > old


def run_full_train_category(
    category: str,
    args: Any,
    runtime: dict[str, Any],
) -> ExperimentResult:
    started = utc_now_iso()
    t0 = time.time()
    out_dir = Path(args.output_dir)
    fail = runtime["fail_on_nonfinite"]
    fg_strategy = tf_adapter.normalize_foreground_strategy(
        getattr(args, "foreground_strategy", "valid_point_plane_fit_with_depth_fallback")
    )
    empty_policy = getattr(args, "empty_gt_pixel_policy", "include_empty_anomalous_masks")
    if empty_policy in {"include", "include_empty"}:
        empty_policy = "include_empty_anomalous_masks"
    elif empty_policy in {"exclude", "exclude_empty"}:
        empty_policy = "exclude_empty_anomalous_masks"

    selection = getattr(args, "checkpoint_selection_metric", DEFAULT_CHECKPOINT_SELECTION)
    if selection not in CHECKPOINT_SELECTION_METRICS:
        raise SystemExit(f"Unknown checkpoint selection metric: {selection}")
    selection_source = getattr(
        args, "checkpoint_selection_source", DEFAULT_CHECKPOINT_SELECTION_SOURCE
    )
    keep_periodic = int(getattr(args, "keep_periodic_checkpoints", -1))
    save_infer = bool(getattr(args, "save_inference_checkpoint", True))
    save_opt = not bool(getattr(args, "no_save_optimizer_checkpoint", False))

    split_info = runtime.get("split_info") or {}
    pre = preflight_full_train(args=args, runtime=runtime, category=category, out_dir=out_dir)
    overlap = pre["overlap"]
    val_eq_test = bool(overlap.get("validation_test_identical"))
    threshold_provenance = (
        "test_optimized_due_to_val_eq_test" if val_eq_test else "validation"
    )

    ckpt_dir = out_dir / "checkpoints" / category
    log_dir = out_dir / "logs" / category
    pred_dir = out_dir / "predictions" / category
    qual_dir = out_dir / "qualitative" / category
    for d in (ckpt_dir, log_dir, pred_dir, qual_dir):
        d.mkdir(parents=True, exist_ok=True)

    tf_adapter.install_plane_mask_patch(strategy=fg_strategy)
    tf_adapter.install_amp_patch(enabled=runtime["amp"])

    ns = tf_adapter.build_namespace(
        data_path=Path(args.root),
        category=category,
        mode=args.mode,
        dtd_path=Path(args.dtd_path),
        seed=args.seed,
    )
    train_ds = tf_adapter.load_train_dataset(ns, foreground_strategy=fg_strategy)
    test_ds = tf_adapter.load_test_dataset(ns, foreground_strategy=fg_strategy)
    if len(train_ds) < 1:
        raise RuntimeError(f"Empty train set: {category}")

    s0 = train_ds[0]
    image = s0["image"]
    plane = s0.get("plane_mask")
    assert_finite("train_image0", image, enabled=fail)
    if plane is None or float(plane.sum()) <= 0:
        raise RuntimeError(f"plane_mask empty for {category} under strategy={fg_strategy}")
    if image.ndim != 3:
        raise RuntimeError(f"unexpected image ndim {image.ndim}")

    net = tf_adapter.load_model(ns)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=float(ns.lr))
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=int(ns.step_size), gamma=float(ns.gamma)
    )
    scaler = None

    x = image.unsqueeze(0).cuda()
    plane_b = plane.unsqueeze(0).cuda()
    idx = torch.tensor([0], device=x.device)
    opt.zero_grad(set_to_none=True)
    loss0, *parts = net.loss(x, idx=idx, plane_mask=plane_b, return_loss=True)
    assert_finite_many(
        {"loss": loss0, "anom": parts[0], "mask": parts[1], "normal": parts[2], "img": parts[3]},
        enabled=fail,
    )
    loss0.backward()
    assert_finite("preflight_grad_norm", torch.tensor(_grad_norm(net)), enabled=fail)
    opt.step()
    for p in net.parameters():
        assert_finite("preflight_param", p, enabled=fail)
    net = tf_adapter.load_model(ns)
    opt = torch.optim.Adam(net.parameters(), lr=float(ns.lr))
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=int(ns.step_size), gamma=float(ns.gamma)
    )

    start_epoch = 0
    global_step = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    threshold_val: float | None = None
    last_ckpt = ckpt_dir / "last.ckpt"
    if getattr(args, "resume", False) and last_ckpt.is_file() and not getattr(args, "force", False):
        loaded = load_training_checkpoint(
            last_ckpt, model=net, optimizer=opt, scheduler=sched, map_location="cuda"
        )
        start_epoch = int(loaded["epoch"]) + 1
        global_step = int(loaded["global_step"])
        best_metric = loaded.get("best_validation_metric")
        best_epoch = loaded.get("extras", {}).get("checkpoint_selection_epoch")
        threshold_val = loaded.get("threshold_selected_from_validation")
        logger.info("Resumed from %s epoch=%s step=%s", last_ckpt, start_epoch - 1, global_step)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.train_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    epochs = int(args.epochs)
    ckpt_every = int(getattr(args, "checkpoint_every", 5))
    val_every = int(getattr(args, "validate_every", 5))
    max_train_steps = getattr(args, "max_train_steps", None)
    max_val_steps = getattr(args, "max_val_steps", None)
    max_test_steps = getattr(args, "max_test_steps", None)
    peak_mem = 0.0
    epoch_times: list[float] = []
    last_metrics: dict[str, Any] = {}
    last_train_loss: float | None = None
    val_paths_ref: list[str] | None = None
    fallback_rate = float(getattr(runtime, "fallback_rate", 0.0) or 0.0)
    if isinstance(runtime, dict):
        fallback_rate = float(runtime.get("foreground_fallback_rate", 0.0) or 0.0)

    config_snapshot = {
        "mode": args.mode,
        "epochs": epochs,
        "seed": args.seed,
        "amp": runtime["amp"],
        "foreground_strategy": fg_strategy,
        "empty_gt_pixel_policy": empty_policy,
        "protocol": runtime["protocol"],
        "train_batch_size": args.train_batch_size,
        "lr": float(ns.lr),
        "step_size": int(ns.step_size),
        "gamma": float(ns.gamma),
        "checkpoint_selection_metric": selection,
        "checkpoint_selection_source": selection_source,
        "eval_kernel_size": int(getattr(args, "eval_kernel_size", 7)),
        "eval_w": float(getattr(args, "eval_w", 0.95)),
        "image_score": "anomaly_map.max()",
        "rgb_depth_fusion": "w*mask_disc + (1-w)*mask_recon",
    }

    try:
        for epoch in range(start_epoch, epochs):
            et0 = time.time()
            net.train()
            running = 0.0
            n_seen = 0
            loss_parts_acc = {"anomaly": 0.0, "mask": 0.0, "normal": 0.0, "img": 0.0}
            plane_ratios: list[float] = []
            synth_ratios: list[float] = []
            gnorms: list[float] = []
            pnorms: list[float] = []
            step_in_epoch = 0

            for data in train_loader:
                if max_train_steps is not None and step_in_epoch >= int(max_train_steps):
                    break
                inputs = data["image"].cuda()
                plane_mask = data["plane_mask"].cuda() if "plane_mask" in data else None
                if plane_mask is not None:
                    plane_ratios.append(float(plane_mask.float().mean().cpu()))
                opt.zero_grad(set_to_none=True)
                loss, anom, mask_l, normal_l, img_l = net.loss(
                    inputs, idx=data["index"].cuda(), plane_mask=plane_mask, return_loss=True
                )
                assert_finite_many(
                    {
                        "loss": loss,
                        "loss_anomaly": anom,
                        "loss_mask": mask_l,
                        "loss_normal": normal_l,
                        "loss_img": img_l,
                    },
                    enabled=fail,
                    context={"epoch": epoch, "step": global_step},
                )
                loss.backward()
                gn = _grad_norm(net)
                assert_finite("grad_norm", torch.tensor(gn), enabled=fail)
                opt.step()
                pn = _param_norm(net)
                assert_finite("param_norm", torch.tensor(pn), enabled=fail)

                bs = int(inputs.shape[0])
                running += float(loss.detach().cpu()) * bs
                n_seen += bs
                loss_parts_acc["anomaly"] += float(anom.detach().cpu()) * bs
                loss_parts_acc["mask"] += float(mask_l.detach().cpu()) * bs
                loss_parts_acc["normal"] += float(normal_l.detach().cpu()) * bs
                loss_parts_acc["img"] += float(img_l.detach().cpu()) * bs
                gnorms.append(gn)
                pnorms.append(pn)
                global_step += 1
                step_in_epoch += 1
                if torch.cuda.is_available():
                    peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024**2))

            sched.step()
            epoch_time = time.time() - et0
            epoch_times.append(epoch_time)
            avg_loss = running / max(n_seen, 1)
            last_train_loss = avg_loss
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "loss_total": avg_loss,
                "loss_anomaly": loss_parts_acc["anomaly"] / max(n_seen, 1),
                "loss_mask": loss_parts_acc["mask"] / max(n_seen, 1),
                "loss_normal": loss_parts_acc["normal"] / max(n_seen, 1),
                "loss_img": loss_parts_acc["img"] / max(n_seen, 1),
                "lr": float(sched.get_last_lr()[0]),
                "grad_norm_mean": float(np.mean(gnorms)) if gnorms else 0.0,
                "param_norm_mean": float(np.mean(pnorms)) if pnorms else 0.0,
                "plane_mask_ratio_mean": float(np.mean(plane_ratios)) if plane_ratios else None,
                "synthetic_mask_ratio_mean": float(np.mean(synth_ratios)) if synth_ratios else None,
                "epoch_time_sec": round(epoch_time, 3),
                "peak_gpu_memory_mb": round(peak_mem, 2),
                "finite_ok": True,
            }
            _append_csv(log_dir / "train_metrics.csv", row)
            _append_jsonl(log_dir / "events.jsonl", {"type": "epoch", **row})

            do_periodic = (epoch % ckpt_every == 0) or (epoch == epochs - 1)
            do_val = (epoch % val_every == 0) or (epoch == epochs - 1)
            is_best = False
            if do_val:
                # Validation only during training. On anomalib_subset val==test.
                maps, gts, labels, paths = _predict_maps(
                    net,
                    test_ds,
                    max_steps=max_val_steps if max_val_steps is not None else max_test_steps,
                    kernel_size=int(getattr(args, "eval_kernel_size", 7)),
                    weight=float(getattr(args, "eval_w", 0.95)),
                    fail=fail,
                    seed=int(args.seed),
                )
                if val_paths_ref is None:
                    val_paths_ref = list(paths)
                    _append_csv(
                        log_dir / "validation_samples.csv",
                        {"n": len(paths), "paths_sha1_prefix": ",".join(Path(p).name for p in paths[:8])},
                    )
                    with (log_dir / "validation_sample_paths.txt").open("w", encoding="utf-8") as f:
                        f.write("\n".join(paths))
                elif paths != val_paths_ref:
                    raise RuntimeError(
                        f"Validation sample order/count changed at epoch {epoch}: "
                        f"{len(paths)} vs {len(val_paths_ref)}"
                    )
                metrics = compute_eval_metrics(
                    maps,
                    gts,
                    labels,
                    empty_policy=empty_policy,
                    aupro_fpr_limit=float(
                        runtime["cfg"].get("evaluation", {}).get("aupro_fpr_limit", 0.3)
                    ),
                    threshold_provenance=threshold_provenance,
                )
                metrics["epoch"] = epoch
                metrics["n_samples"] = len(paths)
                last_metrics = metrics
                threshold_val = metrics.get("selected_threshold")
                _append_csv(
                    log_dir / "validation_metrics.csv",
                    {
                        "epoch": epoch,
                        "image_AUROC": metrics.get("image_AUROC"),
                        "pixel_AUROC": metrics.get("pixel_AUROC"),
                        "author_aupro": metrics.get("author_aupro"),
                        "common_aupro": metrics.get("common_aupro"),
                        "image_score_mean_normal": metrics.get("image_score_mean_normal"),
                        "image_score_mean_anomalous": metrics.get("image_score_mean_anomalous"),
                        "anomaly_map_mean": metrics.get("anomaly_map_mean"),
                        "anomaly_map_std": metrics.get("anomaly_map_std"),
                        "anomaly_map_min": metrics.get("anomaly_map_min"),
                        "anomaly_map_max": metrics.get("anomaly_map_max"),
                        "selected_threshold": metrics.get("selected_threshold"),
                        "threshold_provenance": threshold_provenance,
                        "empty_gt_count": metrics.get("empty_gt_count"),
                        "pixel_metric_policy": empty_policy,
                        "n_samples": len(paths),
                    },
                )
                score = _selection_value(selection, metrics, avg_loss)
                if selection == "last":
                    is_best = epoch == epochs - 1
                    best_metric = score if score is not None else avg_loss
                    best_epoch = epoch
                elif score is not None and (
                    best_metric is None or _is_better(selection, float(score), float(best_metric))
                ):
                    best_metric = float(score)
                    best_epoch = epoch
                    is_best = True

                if maps and getattr(args, "save_qualitative", True):
                    try:
                        np.save(qual_dir / f"epoch_{epoch:04d}_map0.npy", maps[0])
                        np.savez_compressed(
                            pred_dir / f"epoch_{epoch:04d}_maps.npz",
                            maps=np.stack(maps),
                            gts=np.stack(gts),
                            labels=np.asarray(labels),
                            paths=np.asarray(paths),
                        )
                        _append_jsonl(
                            log_dir / "events.jsonl",
                            {
                                "type": "val",
                                "epoch": epoch,
                                "map0_std": float(np.std(maps[0])),
                                "map0_unique": int(len(np.unique(maps[0]))),
                                "map0_min": float(np.min(maps[0])),
                                "map0_max": float(np.max(maps[0])),
                                "n_samples": len(paths),
                            },
                        )
                    except Exception as qexc:  # noqa: BLE001
                        logger.warning("qualitative save failed: %s", qexc)

                if save_infer and is_best:
                    infer = build_inference_checkpoint(
                        epoch=epoch,
                        global_step=global_step,
                        model=net,
                        config_snapshot=config_snapshot,
                        manifest_sha256=runtime["manifest_sha256"],
                        git_commit=git_head_sha(REPO_ROOT),
                        source_repo_commit=tf_adapter.UPSTREAM_ADAM_BENCH["commit"],
                        validation_metric_name=selection,
                        validation_metric_value=best_metric,
                        foreground_strategy=fg_strategy,
                        threshold_from_validation=threshold_val,
                    )
                    save_inference_checkpoint(infer, ckpt_dir / "best_inference.ckpt")
                    # Also keep best.ckpt as inference-sized when selection != last
                    save_inference_checkpoint(infer, ckpt_dir / "best.ckpt")

            # Always refresh last training checkpoint; periodic epoch files optional.
            if save_opt and (do_periodic or do_val or epoch == epochs - 1):
                ckpt = build_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    model=net,
                    optimizer=opt,
                    scheduler=sched,
                    scaler=scaler,
                    config_snapshot=config_snapshot,
                    manifest_sha256=runtime["manifest_sha256"],
                    git_commit=git_head_sha(REPO_ROOT),
                    source_repo_commit=tf_adapter.UPSTREAM_ADAM_BENCH["commit"],
                    dtd_manifest_sha256=runtime["dtd"].get("path_manifest_sha256"),
                    foreground_strategy=fg_strategy,
                    best_validation_metric=best_metric,
                    threshold_from_validation=threshold_val,
                    extras={
                        "checkpoint_selection_metric": selection,
                        "checkpoint_selection_epoch": best_epoch,
                        "checkpoint_selection_value": best_metric,
                    },
                    include_optimizer=True,
                )
                save_training_checkpoint(
                    ckpt,
                    ckpt_dir,
                    is_best=False,
                    write_epoch_file=do_periodic,
                    update_last=True,
                    keep_periodic=keep_periodic if keep_periodic >= 0 else None,
                )
            elif not save_opt and do_periodic:
                infer_last = build_inference_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    model=net,
                    config_snapshot=config_snapshot,
                    manifest_sha256=runtime["manifest_sha256"],
                    git_commit=git_head_sha(REPO_ROOT),
                    source_repo_commit=tf_adapter.UPSTREAM_ADAM_BENCH["commit"],
                    validation_metric_name=selection,
                    validation_metric_value=best_metric,
                    foreground_strategy=fg_strategy,
                    threshold_from_validation=threshold_val,
                )
                save_inference_checkpoint(infer_last, ckpt_dir / "last_inference.ckpt")

            if keep_periodic >= 0 and do_periodic:
                prune_periodic_checkpoints(ckpt_dir, keep_periodic)

            logger.info(
                "epoch %s/%s loss=%.4f time=%.1fs peak_mem=%.0fMB best_epoch=%s",
                epoch,
                epochs,
                avg_loss,
                epoch_time,
                peak_mem,
                best_epoch,
            )

        # Final test pass only after training (same loader on anomalib_subset).
        maps_t, gts_t, labels_t, paths_t = _predict_maps(
            net,
            test_ds,
            max_steps=max_test_steps,
            kernel_size=int(getattr(args, "eval_kernel_size", 7)),
            weight=float(getattr(args, "eval_w", 0.95)),
            fail=fail,
            seed=int(args.seed),
        )
        test_metrics = compute_eval_metrics(
            maps_t,
            gts_t,
            labels_t,
            empty_policy=empty_policy,
            aupro_fpr_limit=float(
                runtime["cfg"].get("evaluation", {}).get("aupro_fpr_limit", 0.3)
            ),
            threshold_provenance=threshold_provenance,
        )
        test_metrics["phase"] = "final_test"
        test_metrics["n_samples"] = len(paths_t)
        _append_csv(
            log_dir / "test_metrics.csv",
            {
                "epoch": epochs - 1,
                "image_AUROC": test_metrics.get("image_AUROC"),
                "pixel_AUROC": test_metrics.get("pixel_AUROC"),
                "author_aupro": test_metrics.get("author_aupro"),
                "common_aupro": test_metrics.get("common_aupro"),
                "threshold_provenance": threshold_provenance,
                "evaluation_is_independent": False if val_eq_test else True,
                "paper_protocol_equivalent": False,
            },
        )
        last_metrics = {**last_metrics, "final_test": test_metrics}

    except Exception as exc:  # noqa: BLE001
        atomic_write_json(
            log_dir / "error.json",
            {"error": f"{type(exc).__name__}: {exc}", "category": category, "at": utc_now_iso()},
        )
        raise

    mean_epoch = float(np.mean(epoch_times)) if epoch_times else None
    result = ExperimentResult(
        experiment_id=f"{runtime['protocol']}:transfusion:{category}",
        model="transfusion",
        category=category,
        table="table5",
        protocol=runtime["protocol"],
        status=ResultStatus.COMPLETED.value,
        started_at=started,
        finished_at=utc_now_iso(),
        elapsed_seconds=round(time.time() - t0, 2),
        seed=args.seed,
        dataset_root=str(Path(args.root)),
        split_manifest_sha256=runtime["manifest_sha256"],
        git_commit=git_head_sha(REPO_ROOT),
        external_repo_url=tf_adapter.UPSTREAM_ADAM_BENCH["url"],
        external_repo_commit=tf_adapter.UPSTREAM_ADAM_BENCH["commit"],
        model_config=config_snapshot,
        data_config={
            "manifest": str(runtime["manifest"]).replace("\\", "/"),
            "manifest_sha256": runtime["manifest_sha256"],
            "paper_protocol_equivalent": False,
            "protocol_difference": runtime["cfg"].get("protocol_difference"),
            "split_counts": split_info,
            "validation_equals_test": split_info.get("validation_equals_test"),
            **overlap,
            "threshold_provenance": threshold_provenance,
            "evaluation_is_independent": not val_eq_test,
            "selection_uses_test_equivalent_data": val_eq_test,
            "dtd": runtime["dtd"],
            "foreground_strategy": fg_strategy,
            "foreground_fallback_rate": fallback_rate,
            "empty_gt_pixel_policy": empty_policy,
        },
        metric_config={
            "threshold_selection_source": (
                "test_optimized" if val_eq_test else "validation"
            ),
            "aupro_fpr_limit": runtime["cfg"].get("evaluation", {}).get("aupro_fpr_limit"),
            "author_aupro_vs_common_aupro": True,
            "checkpoint_selection_metric": selection,
            "checkpoint_selection_source": selection_source,
        },
        image_AUROC=(last_metrics.get("final_test") or last_metrics).get("image_AUROC"),
        pixel_AUROC=(last_metrics.get("final_test") or last_metrics).get("pixel_AUROC"),
        AUPRO=(last_metrics.get("final_test") or last_metrics).get("AUPRO"),
        peak_gpu_memory_mb=round(peak_mem, 2),
        notes=(
            "Pilot/full-train via adapter. anomalib_subset has val==test; "
            "not paper-protocol equivalent. Author checkpoint policy is last-eval overwrite; "
            f"this run selection={selection} ({selection_source})."
        ),
        extras={
            "mean_epoch_time_sec": mean_epoch,
            "last_metrics": last_metrics,
            "estimated_1501_epoch_hours": (mean_epoch * 1501 / 3600.0) if mean_epoch else None,
            "author_aupro": (last_metrics.get("final_test") or last_metrics).get("author_aupro"),
            "common_aupro": (last_metrics.get("final_test") or last_metrics).get("common_aupro"),
            "checkpoint_selection_metric": selection,
            "checkpoint_selection_epoch": best_epoch,
            "checkpoint_selection_value": best_metric,
            "validation_test_identical": val_eq_test,
            "evaluation_is_independent": not val_eq_test,
            "selection_uses_test_equivalent_data": val_eq_test,
            "paper_protocol_equivalent": False,
            "pixel_metric_meta": {
                k: (last_metrics.get("final_test") or last_metrics).get(k)
                for k in (
                    "total_test_images",
                    "empty_gt_count",
                    "included_empty_gt_count",
                    "excluded_empty_gt_count",
                    "pixel_metric_policy",
                )
            },
            "last_train_loss": last_train_loss,
        },
    )
    raw = out_dir / "raw" / "transfusion"
    raw.mkdir(parents=True, exist_ok=True)
    atomic_write_json(raw / f"{category}.json", result.to_dict())
    return result
