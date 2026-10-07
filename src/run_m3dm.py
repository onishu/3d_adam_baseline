"""Run Tencent M3DM (RGB+3D hybrid fusion) on anomalib ADAM3D."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .categories import CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
M3DM_ROOT = ROOT / "third_party" / "AnomalyDetection-M3DM"


def parse_args():
    p = argparse.ArgumentParser(description="Run M3DM on 3D-ADAM")
    p.add_argument("--root", default="datasets/ADAM3D")
    p.add_argument("--categories", nargs="*")
    p.add_argument("--out", default="results_m3dm")
    p.add_argument(
        "--method-name",
        default="DINO+Point_MAE",
        choices=[
            "DINO",
            "Point_MAE",
            "Fusion",
            "DINO+Point_MAE",
            "DINO+Point_MAE+Fusion",
            "DINO+Point_MAE+add",
        ],
    )
    p.add_argument("--xyz-backbone-name", default="Point_MAE")
    p.add_argument(
        "--checkpoint",
        default="third_party/AnomalyDetection-M3DM/checkpoints/pointmae_pretrain.pth",
        help="PointMAE / PointBERT checkpoint path (required for XYZ branches)",
    )
    p.add_argument("--fusion-module-path", default="")
    p.add_argument("--use-uff", action="store_true")
    p.add_argument("--img-size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()
    if not M3DM_ROOT.is_dir():
        raise SystemExit(f"Missing M3DM sources: {M3DM_ROOT}")
    categories = args.categories or CATEGORIES
    unknown = sorted(set(categories) - set(CATEGORIES))
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}")

    out = Path(args.out)
    raw_dir = out / "raw" / "m3dm"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "_env.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "method_name": args.method_name,
                "xyz_backbone_name": args.xyz_backbone_name,
                "checkpoint": args.checkpoint,
                "note": (
                    "Requires PointMAE weights for XYZ/Fusion methods. "
                    "Place checkpoint at --checkpoint. "
                    "See third_party/AnomalyDetection-M3DM/README.md"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    needs_xyz = "Point_MAE" in args.method_name or args.method_name in {
        "Point_MAE",
        "Fusion",
    }
    ckpt = Path(args.checkpoint)
    if needs_xyz and not ckpt.is_file():
        raise SystemExit(
            f"Point backbone checkpoint not found: {ckpt}\n"
            "Download Point-MAE weights into that path, then re-run."
        )

    cmd = [
        sys.executable,
        "main.py",
        "--dataset_type",
        "adam3d",
        "--dataset_path",
        str(Path(args.root).resolve()),
        "--method_name",
        args.method_name,
        "--xyz_backbone_name",
        args.xyz_backbone_name,
        "--img_size",
        str(args.img_size),
        "--category",
        *categories,
    ]
    if args.use_uff:
        cmd.append("--use_uff")
    if args.fusion_module_path:
        cmd.extend(["--fusion_module_path", args.fusion_module_path])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(M3DM_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Keep M3DM markdown outputs under our results tree.
    (out / "m3dm_logs").mkdir(parents=True, exist_ok=True)
    print(">", " ".join(cmd), flush=True)
    print(
        "NOTE: M3DM writes markdown summaries under third_party/AnomalyDetection-M3DM/results/. "
        "Per-category JSON harvest is best-effort after the run.",
        flush=True,
    )
    subprocess.run(cmd, cwd=str(M3DM_ROOT), env=env, check=True)
    print("Done. Inspect M3DM logs and copy metrics into", raw_dir)


if __name__ == "__main__":
    main()
