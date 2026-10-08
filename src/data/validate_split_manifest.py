"""Validate split manifests for path integrity and leakage."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from src.common.config import DEFAULT_DATASET_ROOT
from src.common.hashing import sha256_file
from src.common.logging_utils import setup_logger

logger = logging.getLogger("validate_split_manifest")

REQUIRED_FIELDS = (
    "category",
    "part_id",
    "view_id",
    "label",
    "anomaly_type",
    "split",
    "image_path",
    "depth_path",
    "mask_path",
)

ACTIVE_SPLITS = {"train", "validation", "test"}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        missing = [c for c in REQUIRED_FIELDS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Manifest missing columns {missing}: {path}")
        return list(reader)


def _resolve(dataset_root: Path, rel: str) -> Path:
    return (dataset_root / rel).resolve() if rel else Path()


def validate_manifest(
    manifest_path: Path,
    dataset_root: Path,
    *,
    protocol: str = "anomalib",
    strict_leakage: bool | None = None,
    check_mask_content: bool = True,
) -> dict[str, Any]:
    """Validate a manifest. Returns a report dict; raises on fatal issues when configured."""
    rows = load_manifest(manifest_path)
    digest = sha256_file(manifest_path)

    problems: list[str] = []
    warnings: list[str] = []
    dataset_quality_issues: list[str] = []

    # Duplicate image paths within the same active split
    seen_split_path: set[tuple[str, str]] = set()
    active_rows = [r for r in rows if r["split"] in ACTIVE_SPLITS]

    for r in active_rows:
        key = (r["split"], r["image_path"])
        if key in seen_split_path:
            problems.append(f"duplicate_in_split:{r['split']}:{r['image_path']}")
        seen_split_path.add(key)

    # Path existence + stem alignment
    for r in active_rows:
        img = _resolve(dataset_root, r["image_path"])
        depth = _resolve(dataset_root, r["depth_path"])
        if not img.is_file():
            problems.append(f"missing_image:{r['image_path']}")
        if not depth.is_file():
            problems.append(f"missing_depth:{r['depth_path']}")
        if img.is_file() and depth.is_file() and img.stem != depth.stem:
            problems.append(f"stem_mismatch_rgb_depth:{img.name}:{depth.name}")

        if r["label"] == "abnormal":
            if not r["mask_path"]:
                problems.append(f"missing_mask_path:{r['image_path']}")
            else:
                mask = _resolve(dataset_root, r["mask_path"])
                if not mask.is_file():
                    problems.append(f"missing_mask_file:{r['mask_path']}")
                elif img.is_file() and img.stem != mask.stem:
                    problems.append(f"stem_mismatch_rgb_mask:{img.name}:{mask.name}")
                elif check_mask_content and mask.is_file():
                    try:
                        arr = Image.open(mask).convert("L")
                        extrema = arr.getextrema()
                        if extrema is not None and extrema[1] == 0:
                            # Official subset contains some all-zero masks; treat as
                            # dataset quality, not manifest wiring failure.
                            dataset_quality_issues.append(f"empty_mask:{r['mask_path']}")
                    except Exception as exc:
                        problems.append(f"mask_read_error:{r['mask_path']}:{type(exc).__name__}")

    # Test must contain both classes for each category with active test rows
    by_cat_split_label: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in active_rows:
        by_cat_split_label[r["category"]][r["split"]][r["label"]] += 1

    category_counts: dict[str, dict[str, int]] = {}
    for cat, splits in sorted(by_cat_split_label.items()):
        category_counts[cat] = {f"{sp}_{lb}": int(cnt) for sp, labels in splits.items() for lb, cnt in labels.items()}
        test_labels = splits.get("test", Counter())
        if test_labels:
            if test_labels.get("good", 0) < 1:
                # anomalib protocol always holds out goods; inferred may put all goods in train
                if protocol == "anomalib":
                    problems.append(f"test_missing_normal:{cat}")
                else:
                    warnings.append(f"test_missing_normal:{cat}")
            if test_labels.get("abnormal", 0) < 1:
                problems.append(f"test_missing_abnormal:{cat}")

    # Part-id leakage among parseable IDs.
    # - val↔test overlap on abnormal parts is always critical for paper/inferred.
    # - train↔eval overlap is critical for paper (global part IDs), but only a
    #   warning on anomalib/inferred subsets where stems are locally renumbered.
    parts_by_split: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    abnormal_parts_by_split: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    unparsed = 0
    for r in active_rows:
        if not r["part_id"]:
            unparsed += 1
            continue
        parts_by_split[r["category"]][r["split"]].add(r["part_id"])
        if r["label"] == "abnormal":
            abnormal_parts_by_split[r["category"]][r["split"]].add(r["part_id"])

    leakage: list[dict[str, Any]] = []
    for cat, splits in parts_by_split.items():
        train_parts = splits.get("train", set())
        for other in ("validation", "test"):
            overlap = sorted(train_parts & splits.get(other, set()))
            for part_id in overlap:
                leakage.append(
                    {
                        "category": cat,
                        "part_id": part_id,
                        "between": ["train", other],
                        "scope": "all_labels",
                    }
                )
        abn = abnormal_parts_by_split[cat]
        for part_id in sorted(abn.get("validation", set()) & abn.get("test", set())):
            leakage.append(
                {
                    "category": cat,
                    "part_id": part_id,
                    "between": ["validation", "test"],
                    "scope": "abnormal_only",
                }
            )

    if strict_leakage is None:
        strict_leakage = protocol in {"paper", "inferred"}

    critical = [
        x
        for x in leakage
        if x["scope"] == "abnormal_only"
        or (protocol == "paper" and x["scope"] == "all_labels")
    ]
    noncritical = [x for x in leakage if x not in critical]

    if critical:
        msg = f"part_id_leakage_count:{len(critical)}"
        if strict_leakage:
            problems.append(msg)
        else:
            warnings.append(msg)
    if noncritical:
        warnings.append(f"part_id_train_eval_overlap_count:{len(noncritical)}")

    report = {
        "manifest": str(manifest_path).replace("\\", "/"),
        "manifest_sha256": digest,
        "protocol": protocol,
        "n_rows_total": len(rows),
        "n_rows_active": len(active_rows),
        "n_unparsed_part_id": unparsed,
        "category_counts": category_counts,
        "part_id_leakage": leakage,
        "problems": problems,
        "warnings": warnings,
        "dataset_quality_issues": sorted(set(dataset_quality_issues)),
        "ok": len(problems) == 0,
    }
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a split manifest")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--protocol", default="anomalib")
    p.add_argument("--strict-leakage", action="store_true")
    p.add_argument("--no-strict-leakage", action="store_true")
    p.add_argument("--skip-mask-content", action="store_true")
    p.add_argument(
        "--fail-on-empty-masks",
        action="store_true",
        help="Treat empty GT masks as fatal problems (default: dataset_quality_issues).",
    )
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    global logger
    args = parse_args()
    logger = setup_logger("validate_split_manifest")
    strict: bool | None
    if args.strict_leakage:
        strict = True
    elif args.no_strict_leakage:
        strict = False
    else:
        strict = None

    report = validate_manifest(
        args.manifest,
        Path(args.root),
        protocol=args.protocol,
        strict_leakage=strict,
        check_mask_content=not args.skip_mask_content,
    )
    if args.fail_on_empty_masks and report["dataset_quality_issues"]:
        report["problems"].extend(report["dataset_quality_issues"])
        report["ok"] = False
    out = args.out or Path(args.manifest).with_suffix(".validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Validation ok=%s problems=%d warnings=%d quality=%d",
        report["ok"],
        len(report["problems"]),
        len(report["warnings"]),
        len(report["dataset_quality_issues"]),
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "wrote": str(out),
                "manifest_sha256": report["manifest_sha256"],
                "empty_masks": len(report["dataset_quality_issues"]),
            },
            indent=2,
        )
    )
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
