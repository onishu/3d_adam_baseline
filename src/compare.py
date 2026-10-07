import argparse
import csv
import json
from pathlib import Path
from .categories import CATEGORIES

METRICS_2D = ("image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score")
METRICS_3D = ("image_AUROC", "pixel_AUROC", "AUPRO")
PAPER = {
    "cfa": {"image_AUROC": 79.4, "image_F1Score": 93.9, "pixel_AUROC": 87.0, "pixel_F1Score": 5.8},
    "padim": {"image_AUROC": 96.4, "image_F1Score": 97.5, "pixel_AUROC": 84.6, "pixel_F1Score": 4.1},
    "patchcore": {"image_AUROC": 97.6, "image_F1Score": 97.7, "pixel_AUROC": 84.2, "pixel_F1Score": 6.1},
    "transfusion": {"image_AUROC": 63.3, "pixel_AUROC": 64.4, "AUPRO": 31.0},
    "rgbd_fusion": {},
    "m3dm": {},
    "reg3dad": {},
}
MODEL_METRICS = {
    "cfa": METRICS_2D,
    "padim": METRICS_2D,
    "patchcore": METRICS_2D,
    "transfusion": METRICS_3D,
    "rgbd_fusion": ("image_AUROC", "pixel_AUROC", "image_F1Score", "pixel_F1Score"),
    "m3dm": METRICS_3D,
    "reg3dad": METRICS_3D,
}


def summarize(model, out):
    raw = out / "raw" / model
    metrics = MODEL_METRICS.get(model, METRICS_2D)
    rows = []
    for category in CATEGORIES:
        path = raw / f"{category}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("metrics"):
                rows.append(record)
    if not rows:
        print(f"[{model}] no results in {raw}")
        return
    with (out / f"per_category_{model}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category", *metrics, "elapsed_sec"])
        for row in rows:
            w.writerow(
                [row["category"], *[row["metrics"].get(k, "") for k in metrics], row.get("elapsed_sec", "")]
            )
    present = [k for k in metrics if all(k in r["metrics"] for r in rows)]
    means = {
        k: round(100 * sum(r["metrics"][k] for r in rows) / len(rows), 2)
        for k in present
    }
    paper = PAPER.get(model, {})
    summary = {
        "model": model,
        "n_categories": len(rows),
        "reproduced_mean_pct": means,
        "paper_table_v_pct": paper,
        "delta_pct": {
            k: round(means[k] - paper[k], 2) for k in means if k in paper
        },
    }
    (out / f"summary_{model}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / f"summary_{model}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "reproduced_%", "paper_%", "delta_%"])
        for k in present:
            paper_v = paper.get(k, "")
            delta = summary["delta_pct"].get(k, "")
            w.writerow([k, means[k], paper_v, delta])
    print(f"\n{model.upper()} ({len(rows)}/23 categories)")
    print(f"{'metric':<16}{'repro %':>10}{'paper %':>10}{'delta':>10}")
    for k in present:
        paper_v = paper.get(k)
        delta = summary["delta_pct"].get(k)
        paper_s = f"{paper_v:10.2f}" if isinstance(paper_v, (int, float)) else f"{'':>10}"
        delta_s = f"{delta:10.2f}" if isinstance(delta, (int, float)) else f"{'':>10}"
        print(f"{k:<16}{means[k]:>10.2f}{paper_s}{delta_s}")
    if len(rows) != 23:
        print("WARNING: partial mean; not directly comparable with Table V.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="all",
        choices=[
            "patchcore",
            "padim",
            "cfa",
            "transfusion",
            "rgbd_fusion",
            "m3dm",
            "reg3dad",
            "all",
            "all3d",
        ],
    )
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.model == "all":
        models = ["patchcore", "padim", "cfa"]
    elif a.model == "all3d":
        models = ["transfusion", "rgbd_fusion", "m3dm", "reg3dad"]
    else:
        models = [a.model]
    for model in models:
        summarize(model, out)


if __name__ == "__main__":
    main()
