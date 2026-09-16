import argparse
import csv
import json
from pathlib import Path
from .categories import CATEGORIES

METRICS = ("image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score")
PAPER = {
    "cfa": {"image_AUROC": 79.4, "image_F1Score": 93.9, "pixel_AUROC": 87.0, "pixel_F1Score": 5.8},
    "padim": {"image_AUROC": 96.4, "image_F1Score": 97.5, "pixel_AUROC": 84.6, "pixel_F1Score": 4.1},
    "patchcore": {"image_AUROC": 97.6, "image_F1Score": 97.7, "pixel_AUROC": 84.2, "pixel_F1Score": 6.1},
}


def summarize(model, out):
    raw = out / "raw" / model
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
        w.writerow(["category", *METRICS, "elapsed_sec"])
        for row in rows:
            w.writerow([row["category"], *[row["metrics"].get(k, "") for k in METRICS], row.get("elapsed_sec", "")])
    means = {k: round(100 * sum(r["metrics"][k] for r in rows) / len(rows), 2) for k in METRICS}
    summary = {
        "model": model, "n_categories": len(rows),
        "reproduced_mean_pct": means, "paper_table_v_pct": PAPER[model],
        "delta_pct": {k: round(means[k] - PAPER[model][k], 2) for k in METRICS},
    }
    (out / f"summary_{model}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / f"summary_{model}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "reproduced_%", "paper_%", "delta_%"])
        for k in METRICS:
            w.writerow([k, means[k], PAPER[model][k], summary["delta_pct"][k]])
    print(f"\n{model.upper()} ({len(rows)}/23 categories)")
    print(f"{'metric':<16}{'repro %':>10}{'paper %':>10}{'delta':>10}")
    for k in METRICS:
        print(f"{k:<16}{means[k]:>10.2f}{PAPER[model][k]:>10.2f}{summary['delta_pct'][k]:>10.2f}")
    if len(rows) != 23:
        print("WARNING: partial mean; not directly comparable with Table V.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all", choices=["patchcore", "padim", "cfa", "all"])
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = ["patchcore", "padim", "cfa"] if a.model == "all" else [a.model]
    for model in models:
        summarize(model, out)


if __name__ == "__main__":
    main()
