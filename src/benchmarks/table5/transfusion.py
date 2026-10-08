"""Table V TransFusion entrypoint (adapter-based smoke + gated full-train)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.categories import CATEGORIES
from src.common.config import DEFAULT_DATASET_ROOT, REPO_ROOT, add_common_experiment_args, load_yaml
from src.common.finite_check import NonFiniteError, assert_finite, assert_finite_many, save_debug_bundle
from src.common.hashing import sha256_file
from src.common.logging_utils import setup_logger
from src.common.result_schema import ExperimentResult, ResultStatus, atomic_write_json, utc_now_iso
from src.common.runner import ExperimentRunner, RunnerConfig
from src.common.tensor_stats import tensor_stats
from src.data.adapters import transfusion as tf_adapter

logger = logging.getLogger("benchmarks.table5.transfusion")

MANIFEST_DEFAULT = REPO_ROOT / "configs" / "splits" / "anomalib_manifest.csv"
CONFIG_DEFAULT = REPO_ROOT / "configs" / "models" / "transfusion.yaml"

# Smoke D GPU loss tolerance (10-run audit 2026-10-08; see docs/transfusion_reproduction.md)
# Observed max |Δ|≈2.9e-2, max rel≈5.5e-3 with identical sample order / init / plane mask.
# 1e-3 fails; 1e-2 covers all runs. Residual gap attributed to CUDA nondeterminism
# inside the first training forward (not DataLoader / seed / mask mismatch).
SMOKE_D_LOSS_ATOL = 1e-2
SMOKE_D_LOSS_RTOL = 1e-2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TransFusion Table V (adapter)")
    add_common_experiment_args(p)
    p.set_defaults(protocol="anomalib_subset", seed=3, train_batch_size=1, eval_batch_size=1)
    p.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    p.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    p.add_argument("--mode", default="rgbd", choices=["rgb", "d", "rgbd"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--amp", action="store_true", help="Enable AMP (upstream default; off here)")
    p.add_argument("--no-amp", action="store_true", help="Disable AMP (default)")
    p.add_argument("--fail-on-nonfinite", action="store_true", default=True)
    p.add_argument("--allow-nonfinite", action="store_true")
    p.add_argument(
        "--full-train",
        action="store_true",
        help="Enable full-train/pilot loop (requires preflight gates)",
    )
    p.add_argument("--dtd-path", type=Path, default=REPO_ROOT / "datasets" / "dtd" / "images")
    p.add_argument("--smoke-steps", type=int, default=3)
    p.add_argument("--eval-kernel-size", type=int, default=7)
    p.add_argument("--eval-w", type=float, default=0.95)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--validate-every", type=int, default=5)
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--max-test-steps", type=int, default=None)
    p.add_argument(
        "--foreground-strategy",
        default="valid_point_plane_fit_with_depth_fallback",
        help="Foreground strategy (aliases: valid-depth-mask, adam3d_valid_point_fit, ...)",
    )
    p.add_argument(
        "--empty-gt-pixel-policy",
        default="include_empty_anomalous_masks",
        choices=[
            "include_empty_anomalous_masks",
            "exclude_empty_anomalous_masks",
            "include",
            "exclude",
        ],
    )
    p.add_argument(
        "--checkpoint-selection-metric",
        default="last",
        choices=[
            "validation_image_auroc",
            "validation_pixel_auroc",
            "validation_author_aupro",
            "validation_total_loss",
            "last",
        ],
        help="Author code saves last-eval only; default=last (source: author_code)",
    )
    p.add_argument(
        "--keep-periodic-checkpoints",
        type=int,
        default=-1,
        help="Keep newest N epoch_XXXX.ckpt (-1=keep all). last.ckpt always retained.",
    )
    p.add_argument(
        "--save-inference-checkpoint",
        action="store_true",
        default=True,
        help="Save best_inference.ckpt (model weights only)",
    )
    p.add_argument(
        "--no-save-inference-checkpoint",
        action="store_false",
        dest="save_inference_checkpoint",
    )
    p.add_argument(
        "--no-save-optimizer-checkpoint",
        action="store_true",
        help="Do not write full training checkpoints (optimizer/RNG)",
    )
    p.add_argument("--save-qualitative", action="store_true", default=True)
    p.add_argument("--no-save-qualitative", action="store_false", dest="save_qualitative")
    p.add_argument(
        "--allow-long-train",
        action="store_true",
        help="Required together with --full-train when epochs>=1501 or categories>1 without pilot output",
    )
    return p.parse_args()


def _protocol_name(name: str) -> str:
    aliases = {
        "anomalib": "anomalib_subset",
        "anomalib_subset": "anomalib_subset",
        "anomalib_same_as_test": "anomalib_same_as_test",
        "disk_validation": "disk_validation",
        "inferred_part_split": "inferred_part_split",
        "inferred": "inferred_part_split",
    }
    if name == "paper":
        raise SystemExit(
            "paper protocol unavailable on anomalib subset; use --protocol anomalib_subset "
            "or investigate full HF dataset (docs/full_dataset_protocol.md)"
        )
    if name not in aliases:
        raise SystemExit(f"Unknown protocol: {name}")
    return aliases[name]


def _default_output_dir(protocol: str) -> Path:
    mapping = {
        "anomalib_subset": REPO_ROOT / "results" / "anomalib_protocol" / "transfusion",
        "anomalib_same_as_test": REPO_ROOT / "results" / "anomalib_same_as_test" / "transfusion",
        "disk_validation": REPO_ROOT / "results" / "disk_validation" / "transfusion",
        "inferred_part_split": REPO_ROOT / "results" / "inferred_protocol" / "transfusion",
    }
    return mapping.get(protocol, REPO_ROOT / "results" / protocol / "transfusion")


def _manifest_for_protocol(protocol: str, explicit: Path) -> Path:
    if protocol in {"anomalib_subset", "anomalib_same_as_test"}:
        return explicit
    if protocol == "inferred_part_split":
        return REPO_ROOT / "configs" / "splits" / "inferred_manifest.csv"
    if protocol == "disk_validation":
        # Uses on-disk validation/ paths recorded in anomalib manifest where present;
        # not claimed as paper protocol.
        return explicit
    return explicit


def _manifest_split_counts(manifest: Path, category: str) -> dict[str, Any]:
    import csv

    counts = {"train": 0, "validation": 0, "test": 0}
    labels: dict[str, dict[str, int]] = {"train": {}, "validation": {}, "test": {}}
    with manifest.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["category"] != category:
                continue
            if row["split"] not in counts:
                continue
            counts[row["split"]] += 1
            labels[row["split"]][row["label"]] = labels[row["split"]].get(row["label"], 0) + 1
    return {
        "counts": counts,
        "labels": labels,
        "validation_equals_test": counts["validation"] == counts["test"],
    }


def _prepare_runtime(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = _protocol_name(args.protocol)
    amp = bool(args.amp) and not bool(args.no_amp)
    fail = not bool(args.allow_nonfinite)

    min_dtd = 1 if args.smoke and not args.full_train else 100
    dtd = tf_adapter.validate_dtd(Path(args.dtd_path), min_images=min_dtd)
    if not dtd["ok"]:
        raise SystemExit(
            f"DTD incomplete at {args.dtd_path}: n_images={dtd['n_images']}. "
            "Download full DTD (dtd-r1.0.1) into datasets/dtd/images."
        )

    manifest = _manifest_for_protocol(protocol, Path(args.manifest))
    if not manifest.is_file():
        raise SystemExit(f"Missing manifest: {manifest}")
    man_sha = sha256_file(manifest)

    args.foreground_strategy = tf_adapter.normalize_foreground_strategy(
        args.foreground_strategy
    )
    tf_adapter.ensure_tf_on_path()
    tf_adapter.install_plane_mask_patch(strategy=args.foreground_strategy)
    tf_adapter.install_amp_patch(enabled=amp)

    return {
        "protocol": protocol,
        "amp": amp,
        "fail_on_nonfinite": fail,
        "manifest": manifest,
        "manifest_sha256": man_sha,
        "dtd": dtd,
        "cfg": cfg,
        "paper_protocol_equivalent": False,
        "threshold_selection_source": "n/a",
    }


def _debug_dir(category: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "results" / "debug" / "transfusion" / category / ts


def run_smoke_category(
    category: str,
    args: argparse.Namespace,
    runtime: dict[str, Any],
) -> ExperimentResult:
    started = utc_now_iso()
    t0 = time.time()
    fail = runtime["fail_on_nonfinite"]
    split_info = _manifest_split_counts(runtime["manifest"], category)

    ns = tf_adapter.build_namespace(
        data_path=Path(args.root),
        category=category,
        mode=args.mode,
        dtd_path=Path(args.dtd_path),
        seed=args.seed,
    )
    train = tf_adapter.load_train_dataset(ns, foreground_strategy=args.foreground_strategy)
    if len(train) < 1:
        raise RuntimeError(f"Empty train set for {category}")

    sample = train[0]
    image = sample["image"]
    plane = sample.get("plane_mask")
    assert_finite("dataset_image", image, enabled=fail, context={"category": category})
    if plane is not None:
        assert_finite("dataset_plane_mask", plane, enabled=fail)
        if float(plane.sum()) <= 0:
            raise RuntimeError(
                f"plane_mask is all-zero for {category} after patch; refusing to train"
            )

    net = tf_adapter.load_model(ns)
    net.train()
    x = image.unsqueeze(0).cuda()
    plane_b = plane.unsqueeze(0).cuda() if plane is not None else None
    idx = torch.tensor([0], device=x.device)

    opt = torch.optim.Adam(net.parameters(), lr=float(ns.lr))
    losses: list[float] = []
    peak_mem = 0.0
    for step in range(max(1, int(args.smoke_steps))):
        opt.zero_grad(set_to_none=True)
        try:
            loss, anom, mask_l, normal_l, img_l = net.loss(
                x, idx=idx, plane_mask=plane_b, return_loss=True
            )
            assert_finite_many(
                {
                    "loss_total": loss,
                    "loss_anomaly": anom,
                    "loss_mask": mask_l,
                    "loss_normal": normal_l,
                    "loss_img": img_l,
                },
                enabled=fail,
                context={"category": category, "step": step},
            )
            loss.backward()
            for name, p in net.named_parameters():
                if p.grad is not None:
                    assert_finite(f"grad:{name}", p.grad, enabled=fail)
            grads = [p.grad.detach().float().norm() for p in net.parameters() if p.grad is not None]
            gnorm = torch.stack(grads).norm() if grads else torch.tensor(0.0)
            assert_finite("grad_norm", gnorm, enabled=fail)
            opt.step()
            for name, p in net.named_parameters():
                assert_finite(f"param:{name}", p, enabled=fail)
            losses.append(float(loss.detach().cpu()))
            if torch.cuda.is_available():
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024**2))
        except NonFiniteError as exc:
            dbg = _debug_dir(category)
            save_debug_bundle(
                dbg,
                failure={"error": str(exc), "category": category, "step": step},
                input_stats={
                    "image": tensor_stats("image", x),
                    "plane_mask": tensor_stats("plane_mask", plane_b) if plane_b is not None else None,
                },
                loss_stats={},
                sample_paths=[f"category={category}", "index=0"],
                config=runtime["cfg"],
                arrays={
                    "rgb": x[0, :3].detach().cpu(),
                    "depth": x[0, 3].detach().cpu().numpy() if x.shape[1] > 3 else None,
                    "foreground_mask": plane_b[0].detach().cpu().numpy() if plane_b is not None else None,
                },
            )
            raise

    ckpt_dir = Path(args.output_dir) / "work" / category
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "smoke_model.pkl"
    torch.save(net.state_dict(), ckpt_path)
    net2 = tf_adapter.load_model(ns)
    net2.load_state_dict(torch.load(ckpt_path, map_location="cuda", weights_only=True))
    net2.eval()
    with torch.no_grad():
        b, c, h, w = x.shape
        mask_c = torch.zeros((b, 1, h, w), device=x.device)
        inp = torch.cat([x, mask_c], dim=1)
        t = torch.ones((b,), device=x.device, dtype=torch.long)
        pred_noise, pred_mean, pred_mask = net2.model(inp, t)
        assert_finite_many(
            {"pred_noise": pred_noise, "pred_mean": pred_mean, "pred_mask": pred_mask},
            enabled=fail,
        )
        anomaly_map = torch.sigmoid(pred_mask)
        if anomaly_map.shape[-2:] != x.shape[-2:]:
            raise RuntimeError(
                f"anomaly map spatial {tuple(anomaly_map.shape[-2:])} != image {tuple(x.shape[-2:])}"
            )

    return ExperimentResult(
        experiment_id=f"{runtime['protocol']}:transfusion:{category}",
        model="transfusion",
        category=category,
        table="table5",
        protocol=runtime["protocol"],
        status=ResultStatus.SMOKE.value,
        started_at=started,
        finished_at=utc_now_iso(),
        elapsed_seconds=round(time.time() - t0, 2),
        seed=args.seed,
        dataset_root=str(Path(args.root)),
        split_manifest_sha256=runtime["manifest_sha256"],
        external_repo_url=tf_adapter.UPSTREAM_ADAM_BENCH["url"],
        external_repo_commit=tf_adapter.UPSTREAM_ADAM_BENCH["commit"],
        model_config={
            "mode": args.mode,
            "amp": runtime["amp"],
            "smoke_steps": args.smoke_steps,
            "losses": losses,
            "upstream_spatial_hw": list(x.shape[-2:]),
            "anomaly_map_shape": list(anomaly_map.shape),
            "foreground_strategy": args.foreground_strategy,
        },
        data_config={
            "manifest": str(runtime["manifest"]).replace("\\", "/"),
            "manifest_sha256": runtime["manifest_sha256"],
            "paper_protocol_equivalent": False,
            "protocol_difference": runtime["cfg"].get("protocol_difference"),
            "split_counts": split_info,
            "validation_equals_test": split_info["validation_equals_test"],
            "dtd": runtime["dtd"],
            "plane_mask_sum": float(plane.sum()) if plane is not None else None,
            "foreground_strategy": args.foreground_strategy,
        },
        metric_config={"threshold_selection_source": "n/a_smoke"},
        peak_gpu_memory_mb=round(peak_mem, 2),
        notes="Smoke only",
    )


def run_seed_consistency(args: argparse.Namespace, runtime: dict[str, Any], category: str) -> dict[str, Any]:
    """Smoke D: sample order exact; GPU first-loss within atol/rtol."""
    import random

    losses: list[float] = []
    stems: list[list[str]] = []
    for _trial in range(2):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        ns = tf_adapter.build_namespace(
            data_path=Path(args.root),
            category=category,
            mode=args.mode,
            dtd_path=Path(args.dtd_path),
            seed=args.seed,
        )
        train = tf_adapter.load_train_dataset(ns, foreground_strategy=args.foreground_strategy)
        inner = train
        while hasattr(inner, "dataset"):
            inner = inner.dataset
        files = getattr(inner, "_files", [])
        stems.append([Path(p).stem for p in list(files)[:5]])
        net = tf_adapter.load_model(ns)
        net.train()
        s = train[0]
        x = s["image"].unsqueeze(0).cuda()
        plane = s["plane_mask"].unsqueeze(0).cuda()
        idx = torch.tensor([0], device=x.device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        loss = net.loss(x, idx=idx, plane_mask=plane)
        if not torch.isfinite(loss):
            raise RuntimeError("Smoke D produced non-finite loss")
        losses.append(float(loss.detach().cpu()))
        del net
        torch.cuda.empty_cache()

    sample_order_match = stems[0] == stems[1]
    abs_diff = abs(losses[0] - losses[1])
    rel_diff = abs_diff / max(abs(losses[0]), abs(losses[1]), 1e-12)
    loss_match = abs_diff <= SMOKE_D_LOSS_ATOL or rel_diff <= SMOKE_D_LOSS_RTOL
    return {
        "losses": losses,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "atol": SMOKE_D_LOSS_ATOL,
        "rtol": SMOKE_D_LOSS_RTOL,
        "sample_stems": stems,
        "sample_order_match": sample_order_match,
        "loss_match": loss_match,
        "match": sample_order_match and loss_match,
        "criterion": "sample_order exact; GPU loss atol|rtol (CUDA nondeterminism)",
    }


def _resolve_paths(args: argparse.Namespace) -> None:
    root = Path(args.root)
    args.root = (REPO_ROOT / root).resolve() if not root.is_absolute() else root.resolve()
    args.dtd_path = Path(args.dtd_path)
    if not args.dtd_path.is_absolute():
        args.dtd_path = (REPO_ROOT / args.dtd_path).resolve()
    args.manifest = Path(args.manifest)
    if not args.manifest.is_absolute():
        args.manifest = (REPO_ROOT / args.manifest).resolve()


def main() -> None:
    args = parse_args()
    global logger
    _resolve_paths(args)

    protocol = _protocol_name(args.protocol)
    args.protocol = protocol
    out_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(protocol)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    args.output_dir = out_dir
    logger = setup_logger("benchmarks.table5.transfusion", out_dir / "logs" / "transfusion.log")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = load_yaml(cfg_path)
    runtime = _prepare_runtime(args, cfg)

    categories = list(args.categories) if args.categories else list(CATEGORIES)
    if args.smoke and not args.full_train:
        categories = categories[:1]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for TransFusion smoke/train")

    if args.dry_run:
        runner = ExperimentRunner(
            RunnerConfig(
                model="transfusion",
                table="table5",
                protocol=runtime["protocol"],
                output_dir=out_dir,
                categories=categories,
                seed=args.seed,
                force=args.force,
                resume=args.resume,
                dry_run=True,
                smoke=args.smoke,
                dataset_root=str(args.root),
                split_manifest_sha256=runtime["manifest_sha256"],
            )
        )
        print(
            json.dumps(
                runner.run(
                    lambda c: ExperimentResult(
                        experiment_id=f"dry:{c}",
                        model="transfusion",
                        category=c,
                        table="table5",
                        protocol=runtime["protocol"],
                        status="dry_run",
                        seed=args.seed,
                    )
                ),
                indent=2,
            )
        )
        return

    # Full-train path
    if args.full_train:
        epochs = int(args.epochs) if args.epochs is not None else int(
            cfg.get("training", {}).get("epochs", 1501)
        )
        args.epochs = epochs
        # Safety: refuse 23-cat × 1501 without explicit allow-long-train
        long = epochs >= 1501 or (len(categories) > 1 and epochs > 100)
        if long and not args.allow_long_train:
            raise SystemExit(
                "Refusing long full-train (epochs>=1501 or multi-category >100 epochs). "
                "Use a pilot output dir with smaller --epochs, or pass --allow-long-train "
                "after explicit user approval."
            )

        from src.benchmarks.table5.train_loop import run_full_train_category

        results = []
        for cat in categories:
            runtime["split_info"] = _manifest_split_counts(runtime["manifest"], cat)
            result = run_full_train_category(cat, args, runtime)
            results.append(result.to_dict())
            print(json.dumps({"status": result.status, "category": cat, "metrics": {
                "image_AUROC": result.image_AUROC,
                "pixel_AUROC": result.pixel_AUROC,
                "AUPRO": result.AUPRO,
            }, "extras": result.extras}, indent=2))
        return

    # Default: smoke
    if not args.smoke:
        logger.warning("No --smoke/--full-train: defaulting to --smoke")
        args.smoke = True

    cat = categories[0]
    result = run_smoke_category(cat, args, runtime)
    seed_info = run_seed_consistency(args, runtime, cat)
    result.extras["seed_consistency"] = seed_info
    raw = out_dir / "raw" / "transfusion"
    raw.mkdir(parents=True, exist_ok=True)
    atomic_write_json(raw / f"{cat}.json", result.to_dict())
    print(
        json.dumps(
            {
                "status": result.status,
                "category": cat,
                "losses": result.model_config.get("losses"),
                "seed_consistency": seed_info,
                "manifest_sha256": runtime["manifest_sha256"],
                "wrote": str(raw / f"{cat}.json"),
            },
            indent=2,
        )
    )
    if not seed_info.get("sample_order_match", False):
        raise SystemExit(f"Smoke D failed: sample order mismatch: {seed_info}")
    if not seed_info.get("loss_match", False):
        raise SystemExit(f"Smoke D failed: first-loss mismatch: {seed_info}")


if __name__ == "__main__":
    main()
