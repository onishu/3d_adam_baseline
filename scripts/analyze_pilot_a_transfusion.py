"""Audit Pilot A checkpoints: re-eval twice, write analysis CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.benchmarks.table5.checkpoint import load_training_checkpoint
from src.benchmarks.table5.train_loop import _predict_maps, compute_eval_metrics
from src.common.config import REPO_ROOT
from src.common.result_schema import atomic_write_json
from src.data.adapters import transfusion as tf_adapter


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def eval_checkpoint(
    ckpt_path: Path,
    *,
    root: Path,
    category: str,
    dtd: Path,
    seed: int,
    mode: str,
    fg: str,
    empty_policy: str,
    kernel: int,
    weight: float,
) -> dict[str, Any]:
    ns = tf_adapter.build_namespace(
        data_path=root, category=category, mode=mode, dtd_path=dtd, seed=seed
    )
    tf_adapter.install_plane_mask_patch(strategy=fg)
    tf_adapter.install_amp_patch(enabled=False)
    net = tf_adapter.load_model(ns)
    load_training_checkpoint(ckpt_path, model=net, map_location="cuda", restore_rng=False)
    test_ds = tf_adapter.load_test_dataset(ns, foreground_strategy=fg)
    maps, gts, labels, paths = _predict_maps(
        net,
        test_ds,
        max_steps=None,
        kernel_size=kernel,
        weight=weight,
        fail=True,
        seed=seed,
    )
    metrics = compute_eval_metrics(
        maps,
        gts,
        labels,
        empty_policy=empty_policy,
        threshold_provenance="test_optimized_due_to_val_eq_test",
    )
    metrics["n_samples"] = len(paths)
    metrics["sample_names"] = [Path(p).name for p in paths]
    metrics["paths"] = paths
    # BN/Dropout: UNet uses GroupNorm; confirm eval mode
    metrics["model_training"] = bool(net.training)
    metrics["map0_std"] = float(np.std(maps[0])) if maps else None
    del net
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pilot-dir",
        type=Path,
        default=REPO_ROOT / "results" / "pilots" / "transfusion" / "1m1_e10",
    )
    p.add_argument("--root", type=Path, default=REPO_ROOT / "datasets" / "ADAM3D")
    p.add_argument("--category", default="1m1")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--dtd-path", type=Path, default=REPO_ROOT / "datasets" / "dtd" / "images")
    p.add_argument("--mode", default="rgbd")
    p.add_argument(
        "--foreground-strategy",
        default="valid_point_plane_fit_with_depth_fallback",
    )
    p.add_argument(
        "--empty-gt-pixel-policy",
        default="include_empty_anomalous_masks",
    )
    p.add_argument("--eval-kernel-size", type=int, default=7)
    p.add_argument("--eval-w", type=float, default=0.95)
    p.add_argument("--epochs", default="0,5,9")
    p.add_argument("--skip-reeval", action="store_true")
    args = p.parse_args()

    pilot = args.pilot_dir if args.pilot_dir.is_absolute() else (REPO_ROOT / args.pilot_dir)
    analysis = pilot / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    dtd = args.dtd_path if args.dtd_path.is_absolute() else (REPO_ROOT / args.dtd_path).resolve()
    fg = tf_adapter.normalize_foreground_strategy(args.foreground_strategy)

    train_rows = _read_csv(pilot / "logs" / args.category / "train_metrics.csv")
    val_rows = _read_csv(pilot / "logs" / args.category / "validation_metrics.csv")
    val_by_ep = {int(float(r["epoch"])): r for r in val_rows}
    train_by_ep = {int(float(r["epoch"])): r for r in train_rows}

    learning_curve = []
    for r in train_rows:
        ep = int(float(r["epoch"]))
        vr = val_by_ep.get(ep, {})
        learning_curve.append(
            {
                "epoch": ep,
                "loss_total": float(r["loss_total"]),
                "lr": float(r["lr"]),
                "grad_norm_mean": float(r["grad_norm_mean"]),
                "param_norm_mean": float(r["param_norm_mean"]),
                "image_AUROC": float(vr["image_AUROC"]) if vr.get("image_AUROC") else None,
                "pixel_AUROC": float(vr["pixel_AUROC"]) if vr.get("pixel_AUROC") else None,
                "author_AUPRO": float(vr["author_aupro"]) if vr.get("author_aupro") else None,
                "common_AUPRO": float(vr["common_aupro"]) if vr.get("common_aupro") else None,
            }
        )
    _write_csv(analysis / "learning_curve.csv", learning_curve)

    # Score stats from saved predictions
    score_rows = []
    sample_rows = []
    ckpt_metric_rows = []
    epochs = [int(x) for x in args.epochs.split(",") if x.strip() != ""]
    path_sets: dict[int, list[str]] = {}

    for ep in epochs:
        npz = pilot / "predictions" / args.category / f"epoch_{ep:04d}_maps.npz"
        if npz.is_file():
            data = np.load(npz, allow_pickle=True)
            maps = list(data["maps"])
            gts = list(data["gts"])
            if "labels" in data:
                labels = [int(x) for x in data["labels"]]
            else:
                labels = [int(g.max() > 0) for g in gts]
            if "paths" in data:
                paths = [str(x) for x in data["paths"]]
            else:
                paths = [str(i) for i in range(len(maps))]
            path_sets[ep] = paths
            metrics = compute_eval_metrics(
                maps,
                gts,
                labels,
                empty_policy=args.empty_gt_pixel_policy,
                threshold_provenance="test_optimized_due_to_val_eq_test",
            )
            score_rows.append(
                {
                    "epoch": ep,
                    "source": "saved_predictions",
                    "n_samples": len(maps),
                    **{
                        k: metrics.get(k)
                        for k in (
                            "image_AUROC",
                            "pixel_AUROC",
                            "author_aupro",
                            "common_aupro",
                            "image_score_mean_normal",
                            "image_score_mean_anomalous",
                            "anomaly_map_mean",
                            "anomaly_map_std",
                            "anomaly_map_min",
                            "anomaly_map_max",
                            "selected_threshold",
                            "threshold_provenance",
                            "empty_gt_count",
                            "pixel_metric_policy",
                        )
                    },
                }
            )
            if not sample_rows:
                for i, path in enumerate(paths):
                    sample_rows.append(
                        {
                            "index": i,
                            "path": path,
                            "name": Path(path).name,
                            "label": labels[i],
                            "gt_sum": int(np.asarray(gts[i]).sum()),
                        }
                    )

        tr = train_by_ep.get(ep, {})
        vr = val_by_ep.get(ep, {})
        ckpt_metric_rows.append(
            {
                "epoch": ep,
                "total_loss": float(tr["loss_total"]) if tr else None,
                "image_AUROC": float(vr["image_AUROC"]) if vr.get("image_AUROC") else None,
                "pixel_AUROC": float(vr["pixel_AUROC"]) if vr.get("pixel_AUROC") else None,
                "author_AUPRO": float(vr["author_aupro"]) if vr.get("author_aupro") else None,
                "common_AUPRO": float(vr["common_aupro"]) if vr.get("common_aupro") else None,
                "image_score_mean_normal": score_rows[-1].get("image_score_mean_normal")
                if score_rows
                else None,
                "image_score_mean_anomalous": score_rows[-1].get("image_score_mean_anomalous")
                if score_rows
                else None,
                "anomaly_map_mean": score_rows[-1].get("anomaly_map_mean") if score_rows else None,
                "anomaly_map_std": score_rows[-1].get("anomaly_map_std") if score_rows else None,
                "anomaly_map_min": score_rows[-1].get("anomaly_map_min") if score_rows else None,
                "anomaly_map_max": score_rows[-1].get("anomaly_map_max") if score_rows else None,
                "selected_threshold": score_rows[-1].get("selected_threshold")
                if score_rows
                else None,
                "threshold_provenance": "test_optimized_due_to_val_eq_test",
            }
        )

    same_samples = len({tuple(v) for v in path_sets.values()}) <= 1 if path_sets else None

    reeval_rows = []
    if not args.skip_reeval and torch.cuda.is_available():
        for ep in epochs:
            ckpt = pilot / "checkpoints" / args.category / f"epoch_{ep:04d}.ckpt"
            if not ckpt.is_file():
                continue
            m1 = eval_checkpoint(
                ckpt,
                root=root,
                category=args.category,
                dtd=dtd,
                seed=args.seed,
                mode=args.mode,
                fg=fg,
                empty_policy=args.empty_gt_pixel_policy,
                kernel=args.eval_kernel_size,
                weight=args.eval_w,
            )
            m2 = eval_checkpoint(
                ckpt,
                root=root,
                category=args.category,
                dtd=dtd,
                seed=args.seed,
                mode=args.mode,
                fg=fg,
                empty_policy=args.empty_gt_pixel_policy,
                kernel=args.eval_kernel_size,
                weight=args.eval_w,
            )
            keys = ["image_AUROC", "pixel_AUROC", "author_aupro", "common_aupro"]
            match = all(
                (m1.get(k) is None and m2.get(k) is None)
                or (
                    m1.get(k) is not None
                    and m2.get(k) is not None
                    and abs(float(m1[k]) - float(m2[k])) < 1e-9
                )
                for k in keys
            )
            paths_match = m1.get("paths") == m2.get("paths")
            reeval_rows.append(
                {
                    "epoch": ep,
                    "run1_image_AUROC": m1.get("image_AUROC"),
                    "run2_image_AUROC": m2.get("image_AUROC"),
                    "run1_pixel_AUROC": m1.get("pixel_AUROC"),
                    "run2_pixel_AUROC": m2.get("pixel_AUROC"),
                    "run1_author_aupro": m1.get("author_aupro"),
                    "run2_author_aupro": m2.get("author_aupro"),
                    "n_samples": m1.get("n_samples"),
                    "paths_identical": paths_match,
                    "metrics_identical": match,
                    "model_training_after_eval": m1.get("model_training"),
                    "map0_std": m1.get("map0_std"),
                    "image_score_mean_normal": m1.get("image_score_mean_normal"),
                    "image_score_mean_anomalous": m1.get("image_score_mean_anomalous"),
                    "anomaly_map_std": m1.get("anomaly_map_std"),
                    "selected_threshold": m1.get("selected_threshold"),
                }
            )
            # enrich checkpoint_metrics with reeval
            for row in ckpt_metric_rows:
                if int(row["epoch"]) == ep:
                    row["reeval_image_AUROC"] = m1.get("image_AUROC")
                    row["reeval_metrics_identical"] = match
                    row["image_score_mean_normal"] = m1.get("image_score_mean_normal")
                    row["image_score_mean_anomalous"] = m1.get("image_score_mean_anomalous")
                    row["anomaly_map_mean"] = m1.get("anomaly_map_mean")
                    row["anomaly_map_std"] = m1.get("anomaly_map_std")
                    row["anomaly_map_min"] = m1.get("anomaly_map_min")
                    row["anomaly_map_max"] = m1.get("anomaly_map_max")
                    row["selected_threshold"] = m1.get("selected_threshold")

    _write_csv(analysis / "checkpoint_metrics.csv", ckpt_metric_rows)
    _write_csv(analysis / "score_statistics.csv", score_rows)
    _write_csv(analysis / "validation_samples.csv", sample_rows)
    if reeval_rows:
        _write_csv(analysis / "checkpoint_reeval.csv", reeval_rows)

    # Learning curve markdown
    md = [
        "# Pilot A learning curve (1m1, 10 epochs)",
        "",
        "## Observation",
        "",
        "Training loss decreased (≈3.80 → 1.06) while validation detection metrics",
        "worsened after epoch 0 (image AUROC 0.756 → 0.500 at epoch 5, 0.581 at epoch 9).",
        "",
        "## Evaluation protocol (audited)",
        "",
        f"- foreground_strategy: `{fg}`",
        f"- empty_gt_pixel_policy: `{args.empty_gt_pixel_policy}`",
        "- image score: `anomaly_map.max()` (author EvaluationUtils fusion)",
        "- fusion: `0.95 * mask_disc + 0.05 * mask_recon`",
        "- smoothing: box kernel 7×7",
        "- `model.eval()` + `torch.inference_mode()` during eval",
        "- synthetic anomaly generation: train-only (`net.loss`); not used in EvaluationUtils",
        "- validation/test overlap: identical on anomalib_subset",
        f"- saved prediction path sets identical across epochs: `{same_samples}`",
        "",
        "## Curve",
        "",
        "| epoch | loss | image AUROC | pixel AUROC | author AUPRO |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in learning_curve:
        md.append(
            f"| {r['epoch']} | {r['loss_total']:.4f} | "
            f"{r['image_AUROC'] if r['image_AUROC'] is not None else ''} | "
            f"{r['pixel_AUROC'] if r['pixel_AUROC'] is not None else ''} | "
            f"{r['author_AUPRO'] if r['author_AUPRO'] is not None else ''} |"
        )
    md.extend(
        [
            "",
            "## Candidate causes of metric drop",
            "",
            "1. Optimization fits synthetic-anomaly discrimination; real defect scores may not rise.",
            "2. Image score = max(map); if normal maps develop high peaks, AUROC collapses.",
            "3. Diffusion eval uses stochastic sampling unless seeded (this audit seeds eval).",
            "4. Foreground strategy is valid-depth-equivalent (plane-fit fallback); author MVTec plane differs.",
            "5. Best-by-loss would pick last epoch; author checkpoint policy is last-eval overwrite.",
            "",
        ]
    )
    if reeval_rows:
        md.append("## Checkpoint re-evaluation (2×)")
        md.append("")
        for r in reeval_rows:
            md.append(
                f"- epoch {r['epoch']}: metrics_identical={r['metrics_identical']}, "
                f"paths_identical={r['paths_identical']}, "
                f"image_AUROC={r['run1_image_AUROC']}"
            )
        md.append("")
    (analysis / "learning_curve.md").write_text("\n".join(md), encoding="utf-8")

    summary = {
        "same_validation_samples_across_epochs": same_samples,
        "n_validation_samples": len(sample_rows),
        "reeval": reeval_rows,
        "foreground_strategy": fg,
        "empty_gt_pixel_policy": args.empty_gt_pixel_policy,
        "validation_test_identical": True,
        "evaluation_is_independent": False,
        "paper_protocol_equivalent": False,
    }
    atomic_write_json(analysis / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
