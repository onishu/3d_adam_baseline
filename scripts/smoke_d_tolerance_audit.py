"""Run Smoke D first-loss comparison N times; recommend atol/rtol."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.common.config import REPO_ROOT
from src.common.result_schema import atomic_write_json
from src.data.adapters import transfusion as tf_adapter


def one_pair(args: argparse.Namespace) -> dict:
    losses: list[float] = []
    stems: list[list[str]] = []
    masks: list[np.ndarray] = []
    inits: list[float] = []
    for _ in range(2):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        ns = tf_adapter.build_namespace(
            data_path=args.root,
            category=args.category,
            mode=args.mode,
            dtd_path=args.dtd_path,
            seed=args.seed,
        )
        train = tf_adapter.load_train_dataset(ns, foreground_strategy=args.foreground_strategy)
        inner = train
        while hasattr(inner, "dataset"):
            inner = inner.dataset
        files = getattr(inner, "_files", [])
        stems.append([Path(p).stem for p in list(files)[:8]])
        net = tf_adapter.load_model(ns)
        # initial weight fingerprint
        with torch.no_grad():
            w0 = next(net.parameters()).detach().float().reshape(-1)[:64].cpu().numpy()
        inits.append(float(np.linalg.norm(w0)))
        net.train()
        s = train[0]
        x = s["image"].unsqueeze(0).cuda()
        plane = s["plane_mask"].unsqueeze(0).cuda()
        masks.append(plane.detach().cpu().numpy().copy())
        idx = torch.tensor([0], device=x.device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        loss = net.loss(x, idx=idx, plane_mask=plane)
        losses.append(float(loss.detach().cpu()))
        del net
        torch.cuda.empty_cache()

    abs_diff = abs(losses[0] - losses[1])
    rel_diff = abs_diff / max(abs(losses[0]), abs(losses[1]), 1e-12)
    return {
        "loss_a": losses[0],
        "loss_b": losses[1],
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "sample_order_identical": stems[0] == stems[1],
        "synthetic_or_plane_mask_identical": bool(np.allclose(masks[0], masks[1])),
        "model_initial_state_identical": abs(inits[0] - inits[1]) < 1e-12,
        "init_norm_a": inits[0],
        "init_norm_b": inits[1],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=REPO_ROOT / "datasets" / "ADAM3D")
    p.add_argument("--dtd-path", type=Path, default=REPO_ROOT / "datasets" / "dtd" / "images")
    p.add_argument("--category", default="1m1")
    p.add_argument("--mode", default="rgbd")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument(
        "--foreground-strategy",
        default="valid_depth_mask",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "quality" / "smoke_d_tolerance",
    )
    args = p.parse_args()
    args.root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    args.dtd_path = (
        args.dtd_path if args.dtd_path.is_absolute() else (REPO_ROOT / args.dtd_path).resolve()
    )
    args.foreground_strategy = tf_adapter.normalize_foreground_strategy(
        args.foreground_strategy
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    tf_adapter.install_plane_mask_patch(strategy=args.foreground_strategy)
    tf_adapter.install_amp_patch(enabled=False)

    rows = []
    for i in range(args.runs):
        r = one_pair(args)
        r["run_index"] = i
        rows.append(r)
        print(json.dumps(r))

    abs_diffs = [r["abs_diff"] for r in rows]
    rel_diffs = [r["rel_diff"] for r in rows]
    max_abs = float(max(abs_diffs))
    max_rel = float(max(rel_diffs))
    # Recommend tight bounds with small headroom over observed max.
    # Prefer atol=1e-3 / rtol=1e-3 if they cover max*1.5; else escalate.
    cand = [(1e-3, 1e-3), (5e-3, 5e-3), (1e-2, 1e-2), (5e-2, 1e-2)]
    chosen = None
    for atol, rtol in cand:
        ok = all(a <= atol or r <= rtol for a, r in zip(abs_diffs, rel_diffs))
        if ok:
            chosen = {"atol": atol, "rtol": rtol, "covers_all_runs": True}
            break
    if chosen is None:
        chosen = {
            "atol": max(max_abs * 2, 1e-3),
            "rtol": max(max_rel * 2, 1e-3),
            "covers_all_runs": False,
            "note": "exceeded candidates; investigate first differing op",
        }

    summary = {
        "runs": args.runs,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mean_abs_diff": float(np.mean(abs_diffs)),
        "p95_abs_diff": float(np.percentile(abs_diffs, 95)),
        "sample_order_all_identical": all(r["sample_order_identical"] for r in rows),
        "mask_all_identical": all(r["synthetic_or_plane_mask_identical"] for r in rows),
        "init_all_identical": all(r["model_initial_state_identical"] for r in rows),
        "recommended": chosen,
        "candidate_1e-3_pass": all(a <= 1e-3 or r <= 1e-3 for a, r in zip(abs_diffs, rel_diffs)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "smoke_d_10runs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    atomic_write_json(args.out_dir / "smoke_d_10runs_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
