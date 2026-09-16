import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from .categories import CATEGORIES
from .models import MAX_EPOCHS, build_model

SEED = 42
METRICS = ("image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["patchcore", "padim", "cfa"])
    p.add_argument("--root", default="datasets/ADAM3D")
    p.add_argument("--categories", nargs="*")
    p.add_argument("--out", default="results")
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def run_one(args, category):
    from anomalib.data import ADAM3D
    seed_everything(SEED, workers=True)
    dm = ADAM3D(
        root=args.root, category=category,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, seed=SEED,
    )
    model = build_model(args.model)
    trainer = Trainer(
        accelerator="gpu", devices=1,
        max_epochs=MAX_EPOCHS[args.model],
        logger=False, enable_checkpointing=False,
        num_sanity_val_steps=0, enable_model_summary=False,
        default_root_dir=str(Path(args.out) / "work" / args.model / category),
    )
    started = time.time()
    trainer.fit(model=model, datamodule=dm)
    result = trainer.test(model=model, datamodule=dm, verbose=False)[0]
    metrics = {key: float(result[key]) for key in METRICS if key in result}
    return {"model": args.model, "category": category, "metrics": metrics,
            "elapsed_sec": round(time.time() - started, 2)}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU was not detected.")
    categories = args.categories or CATEGORIES
    unknown = sorted(set(categories) - set(CATEGORIES))
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}")
    torch.set_float32_matmul_precision("high")
    raw_dir = Path(args.out) / "raw" / args.model
    raw_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "seed": SEED,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "max_epochs": MAX_EPOCHS[args.model],
    }
    (raw_dir / "_env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    for index, category in enumerate(categories, 1):
        path = raw_dir / f"{category}.json"
        if path.exists() and not args.force:
            print(f"[{index}/{len(categories)}] SKIP {args.model}/{category}")
            continue
        print(f"[{index}/{len(categories)}] RUN {args.model}/{category}", flush=True)
        try:
            record = run_one(args, category)
            print("  " + "  ".join(
                f"{key}={record['metrics'].get(key, float('nan')):.4f}"
                for key in METRICS), flush=True)
        except Exception as exc:
            record = {"model": args.model, "category": category,
                      "error": f"{type(exc).__name__}: {exc}"}
            print("  ERROR:", record["error"], flush=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        gc.collect()
        torch.cuda.empty_cache()
    print("Done:", raw_dir)
    print(f"Next: uv run python -m src.compare --model {args.model}")


if __name__ == "__main__":
    main()
