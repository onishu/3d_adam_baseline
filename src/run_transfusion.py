"""Train/evaluate author TransFusion (RGB+D) on anomalib ADAM3D layout."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

from .categories import CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
TF_ROOT = ROOT / "third_party" / "3D-ADAMBench" / "TransFusion_3D-ADAM"
DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
PAPER = {
    "image_AUROC": 0.633,
    "pixel_AUROC": 0.644,
    "AUPRO": 0.310,
}


def parse_args():
    p = argparse.ArgumentParser(description="Run TransFusion on 3D-ADAM")
    p.add_argument("--root", default="datasets/ADAM3D")
    p.add_argument("--categories", nargs="*")
    p.add_argument("--out", default="results_transfusion")
    p.add_argument("--run-name", default="transfusion_adam3d")
    p.add_argument("--mode", default="rgbd", choices=["rgb", "d", "rgbd"])
    p.add_argument("--dtd-path", default="datasets/dtd/images")
    p.add_argument("--epoch-num", type=int, default=1501)
    p.add_argument("--eval-epoch-num", type=int, default=750)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--download-dtd", action="store_true")
    return p.parse_args()


def ensure_dtd(dtd_images: Path, download: bool) -> None:
    if dtd_images.is_dir() and any(dtd_images.iterdir()):
        return
    if not download:
        raise SystemExit(
            f"DTD textures not found at {dtd_images}. "
            "Download manually or pass --download-dtd."
        )
    archive = ROOT / "datasets" / "dtd-r1.0.1.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print("Downloading DTD...", flush=True)
        urlretrieve(DTD_URL, archive)
    import tarfile

    print("Extracting DTD...", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(ROOT / "datasets")
    if not dtd_images.is_dir():
        raise SystemExit(f"DTD extract failed; expected {dtd_images}")


def run_tf(args_list: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TF_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(TF_ROOT / "Experiment.py"), *args_list]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(TF_ROOT), env=env, check=True)


def read_category_metrics(csv_path: Path, category: str) -> dict[str, float] | None:
    if not csv_path.exists():
        return None
    import pandas as pd

    df = pd.read_csv(csv_path)
    rows = df[df["Category"] == category]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    return {
        "image_AUROC": float(row["Roc Img"]),
        "pixel_AUROC": float(row["Roc Pixel"]),
        "AUPRO": float(row["PRO Img"]),
        "pixel_AP": float(row["AP Pixel"]),
    }


def main():
    args = parse_args()
    if not TF_ROOT.is_dir():
        raise SystemExit(f"Missing TransFusion sources: {TF_ROOT}")
    ensure_dtd(Path(args.dtd_path), args.download_dtd)
    categories = args.categories or CATEGORIES
    unknown = sorted(set(categories) - set(CATEGORIES))
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}")

    out = Path(args.out)
    raw_dir = out / "raw" / "transfusion"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = (TF_ROOT / "experiments").resolve()
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "mode": args.mode,
        "epoch_num": args.epoch_num,
        "seed": args.seed,
        "paper_table_v": PAPER,
        "tf_root": str(TF_ROOT),
    }
    (raw_dir / "_env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    for index, category in enumerate(categories, 1):
        path = raw_dir / f"{category}.json"
        if path.exists() and not args.force:
            print(f"[{index}/{len(categories)}] SKIP transfusion/{category}")
            continue
        print(f"[{index}/{len(categories)}] RUN transfusion/{category}", flush=True)
        started = time.time()
        common = [
            "-r", args.run_name,
            "-d", str(Path(args.root).resolve()),
            "-ds", "adam3d",
            "--mode", args.mode,
            "--dtd-path", str(Path(args.dtd_path).resolve()),
            "--category", category,
            "--epoch-num", str(args.epoch_num),
            "--eval-epoch-num", str(args.eval_epoch_num),
            "--seed", str(args.seed),
            "--log-path", str(log_path) + "/",
            "--no-fg-masks",
        ]
        try:
            if not args.skip_train:
                run_tf(["-c", "train", *common])
            run_tf(["-c", "test", *common])
            metrics = read_category_metrics(
                log_path / args.run_name / "output.csv", category
            )
            if not metrics:
                raise RuntimeError("metrics missing from TransFusion output.csv")
            record = {
                "model": "transfusion",
                "category": category,
                "metrics": metrics,
                "elapsed_sec": round(time.time() - started, 2),
            }
            print(
                "  "
                + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                flush=True,
            )
        except Exception as exc:
            record = {
                "model": "transfusion",
                "category": category,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print("  ERROR:", record["error"], flush=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("Done:", raw_dir)
    print("Next: uv run python -m src.compare --model transfusion --out", args.out)


if __name__ == "__main__":
    main()
