"""Export ADAM3D to PCD and run Reg3D-AD (Real3D-AD benchmark)."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .categories import CATEGORIES
from .export_adam3d_pcd import export_category

ROOT = Path(__file__).resolve().parents[1]
REG_ROOT = ROOT / "third_party" / "Real3D-AD"


def parse_args():
    p = argparse.ArgumentParser(description="Run Reg3D-AD on 3D-ADAM")
    p.add_argument("--root", default="datasets/ADAM3D")
    p.add_argument("--pcd-root", default="datasets/ADAM3D_pcd")
    p.add_argument("--categories", nargs="*")
    p.add_argument("--out", default="results_reg3dad")
    p.add_argument("--max-points", type=int, default=20000)
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--memory-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if not REG_ROOT.is_dir():
        raise SystemExit(f"Missing Real3D-AD sources: {REG_ROOT}")
    categories = args.categories or CATEGORIES
    unknown = sorted(set(categories) - set(CATEGORIES))
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}")

    pcd_root = Path(args.pcd_root)
    if not args.skip_export:
        pcd_root.mkdir(parents=True, exist_ok=True)
        for category in categories:
            n_train, n_test = export_category(
                Path(args.root), pcd_root, category, args.max_points
            )
            print(f"export {category}: train={n_train} test={n_test}", flush=True)

    out = Path(args.out)
    raw_dir = out / "raw" / "reg3dad"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "_env.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "pcd_root": str(pcd_root.resolve()),
                "memory_size": args.memory_size,
                "seed": args.seed,
                "note": (
                    "Reg3D-AD depends on open3d / faiss / PointMAE-style features "
                    "from third_party/Real3D-AD. Install that env separately if needed."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REG_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["ADAM3D_MODE"] = "1"
    env["ADAM3D_PCD_ROOT"] = str(pcd_root.resolve())
    env["ADAM3D_CATEGORIES"] = ",".join(categories)
    env["REG3DAD_OUT"] = str((out / "benchmark").resolve()) + "/"

    cmd = [
        sys.executable,
        "main_reg3dad.py",
        "--gpu",
        "0",
        "--seed",
        str(args.seed),
        "--memory_size",
        str(args.memory_size),
        "--anomaly_scorer_num_nn",
        "1",
        "--faiss_on_gpu",
        "--faiss_num_workers",
        "0",
        "sampler",
        "-p",
        "0.1",
        "approx_greedy_coreset",
    ]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REG_ROOT), env=env, check=True)
    print("Done. Reg3D-AD benchmark outputs under", out / "benchmark")


if __name__ == "__main__":
    main()
