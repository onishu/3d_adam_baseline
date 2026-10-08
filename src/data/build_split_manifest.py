"""Build fixed split manifests for 3D-ADAM reproduction protocols."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch

from src.categories import CATEGORIES
from src.common.config import DEFAULT_DATASET_ROOT, DEFAULT_SEED, REPO_ROOT, load_protocol_registry
from src.common.hashing import sha256_file, sha256_text
from src.common.logging_utils import setup_logger
from src.data.filename_parsing import parse_stem

MANIFEST_FIELDS = (
    "category",
    "part_id",
    "view_id",
    "label",
    "anomaly_type",
    "split",
    "image_path",
    "depth_path",
    "mask_path",
    "stem",
    "parse_status",
    "source_split_dir",
)

logger = logging.getLogger("build_split_manifest")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _depth_for_rgb(rgb: Path) -> Path:
    return rgb.parent.parent / "xyz" / f"{rgb.stem}.tiff"


def _mask_for_rgb(rgb: Path, label: str) -> str:
    if label == "good":
        return ""
    return str((rgb.parent.parent / "ground_truth" / f"{rgb.stem}.png").as_posix())


def holdout_indices(n: int, ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Mirror anomalib random_split lengths + torch.randperm seeding."""
    if n <= 0:
        return [], []
    split_ratio = [1 - ratio, ratio]
    lengths = [math.floor(n * r) for r in split_ratio]
    for i in range(n - sum(lengths)):
        lengths[i % sum(lengths)] += 1
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()
    train_idx = order[: lengths[0]]
    hold_idx = order[lengths[0] : lengths[0] + lengths[1]]
    return sorted(train_idx), sorted(hold_idx)


def _row_from_rgb(
    *,
    category: str,
    dataset_root: Path,
    source_split_dir: str,
    label: str,
    rgb: Path,
    split: str,
) -> dict[str, str]:
    parsed = parse_stem(rgb.stem)
    anomaly_type = "none" if label == "good" else label
    depth = _depth_for_rgb(rgb)
    mask = _mask_for_rgb(rgb, label)
    return {
        "category": category,
        "part_id": parsed.part_id or "",
        "view_id": parsed.view_id or "",
        "label": "good" if label == "good" else "abnormal",
        "anomaly_type": anomaly_type,
        "split": split,
        "image_path": _rel(rgb, dataset_root),
        "depth_path": _rel(depth, dataset_root),
        "mask_path": _rel(Path(mask), dataset_root) if mask else "",
        "stem": rgb.stem,
        "parse_status": parsed.parse_status,
        "source_split_dir": source_split_dir,
    }


def build_anomalib_protocol_rows(
    dataset_root: Path,
    categories: list[str],
    seed: int = DEFAULT_SEED,
    test_split_ratio: float = 0.2,
) -> list[dict[str, str]]:
    """Build rows matching anomalib ADAM3D effective splits.

    Important: on-disk ``validation/`` is unused by anomalib defaults.
    Normals for test/val come from a seed-controlled hold-out of ``train/good``.
    Validation equals test (SAME_AS_TEST).
    """
    rows: list[dict[str, str]] = []
    for category in categories:
        cat_root = dataset_root / category
        if not cat_root.is_dir():
            raise FileNotFoundError(f"Missing category directory: {cat_root}")

        train_goods = sorted((cat_root / "train" / "good" / "rgb").glob("*.png"))
        if not train_goods:
            raise FileNotFoundError(f"No train/good RGB for {category}")

        train_idx, hold_idx = holdout_indices(len(train_goods), test_split_ratio, seed)
        hold_set = set(hold_idx)
        for i, rgb in enumerate(train_goods):
            split = "test" if i in hold_set else "train"
            # SAME_AS_TEST: also emit validation copy for held-out normals and defects later
            rows.append(
                _row_from_rgb(
                    category=category,
                    dataset_root=dataset_root,
                    source_split_dir="train",
                    label="good",
                    rgb=rgb,
                    split=split,
                )
            )
            if split == "test":
                rows.append(
                    _row_from_rgb(
                        category=category,
                        dataset_root=dataset_root,
                        source_split_dir="train",
                        label="good",
                        rgb=rgb,
                        split="validation",
                    )
                )

        # Disk validation/ is unused by anomalib ADAM3D defaults; keep audit rows.
        val_dir = cat_root / "validation"
        if val_dir.is_dir():
            for label_dir in sorted(p for p in val_dir.iterdir() if p.is_dir()):
                rgb_dir = label_dir / "rgb"
                if not rgb_dir.is_dir():
                    continue
                for rgb in sorted(rgb_dir.glob("*.png")):
                    rows.append(
                        _row_from_rgb(
                            category=category,
                            dataset_root=dataset_root,
                            source_split_dir="validation",
                            label=label_dir.name,
                            rgb=rgb,
                            split="unused_on_disk_validation",
                        )
                    )

        test_root = cat_root / "test"
        if test_root.is_dir():
            for label_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
                rgb_dir = label_dir / "rgb"
                if not rgb_dir.is_dir():
                    continue
                for rgb in sorted(rgb_dir.glob("*.png")):
                    for split in ("test", "validation"):
                        rows.append(
                            _row_from_rgb(
                                category=category,
                                dataset_root=dataset_root,
                                source_split_dir="test",
                                label=label_dir.name,
                                rgb=rgb,
                                split=split,
                            )
                        )
    return rows


def build_inferred_protocol_rows(
    dataset_root: Path,
    categories: list[str],
    seed: int = DEFAULT_SEED,
    defective_test_ratio: float = 0.6,
) -> list[dict[str, str]]:
    """Best-effort inferred protocol (NOT paper-official).

    - All defect-free scans (train/good + validation/good) -> train
    - Defect samples with parseable part_id: split part-instances 60/40 test/val
    - Unparseable defect samples: assigned to test with part_id empty
    """
    rows: list[dict[str, str]] = []
    g = torch.Generator().manual_seed(seed)

    for category in categories:
        cat_root = dataset_root / category
        if not cat_root.is_dir():
            raise FileNotFoundError(f"Missing category directory: {cat_root}")

        # All goods -> train
        for source_split in ("train", "validation"):
            good_rgb = cat_root / source_split / "good" / "rgb"
            if not good_rgb.is_dir():
                continue
            for rgb in sorted(good_rgb.glob("*.png")):
                rows.append(
                    _row_from_rgb(
                        category=category,
                        dataset_root=dataset_root,
                        source_split_dir=source_split,
                        label="good",
                        rgb=rgb,
                        split="train",
                    )
                )

        # Defects
        defect_by_part: dict[str, list[tuple[str, Path]]] = {}
        unparsed: list[tuple[str, Path]] = []
        test_root = cat_root / "test"
        if test_root.is_dir():
            for label_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
                if label_dir.name == "good":
                    # if any good under test, keep as test normals
                    for rgb in sorted((label_dir / "rgb").glob("*.png")):
                        rows.append(
                            _row_from_rgb(
                                category=category,
                                dataset_root=dataset_root,
                                source_split_dir="test",
                                label="good",
                                rgb=rgb,
                                split="test",
                            )
                        )
                    continue
                rgb_dir = label_dir / "rgb"
                if not rgb_dir.is_dir():
                    continue
                for rgb in sorted(rgb_dir.glob("*.png")):
                    parsed = parse_stem(rgb.stem)
                    if parsed.part_id is not None:
                        defect_by_part.setdefault(parsed.part_id, []).append((label_dir.name, rgb))
                    else:
                        unparsed.append((label_dir.name, rgb))

        part_ids = sorted(defect_by_part)
        if part_ids:
            n = len(part_ids)
            n_test = max(1, int(round(n * defective_test_ratio)))
            order = torch.randperm(n, generator=g).tolist()
            test_parts = {part_ids[i] for i in order[:n_test]}
            for part_id, items in defect_by_part.items():
                split = "test" if part_id in test_parts else "validation"
                for label, rgb in items:
                    rows.append(
                        _row_from_rgb(
                            category=category,
                            dataset_root=dataset_root,
                            source_split_dir="test",
                            label=label,
                            rgb=rgb,
                            split=split,
                        )
                    )

        for label, rgb in unparsed:
            rows.append(
                _row_from_rgb(
                    category=category,
                    dataset_root=dataset_root,
                    source_split_dir="test",
                    label=label,
                    rgb=rgb,
                    split="test",
                )
            )

    return rows


def build_paper_protocol_rows(
    dataset_root: Path,
    categories: list[str],
) -> list[dict[str, str]]:
    """Paper protocol cannot be reconstructed from anomalib subset alone."""
    raise RuntimeError(
        "paper protocol is blocked on the public anomalib subset: defective stems "
        "often lack part-instance IDs required for the paper's 60:40 part-instance "
        f"split. dataset_root={dataset_root} categories={len(categories)}. "
        "Use --protocol anomalib or --protocol inferred, or provide the full "
        "HuggingFace pmchard/3D-ADAM part-instance layout."
    )


def write_manifest(
    rows: list[dict[str, str]],
    out_csv: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})

    digest = sha256_file(out_csv)
    meta_out = {
        **meta,
        "n_rows": len(rows),
        "manifest_csv": str(out_csv).replace("\\", "/"),
        "manifest_sha256": digest,
        "manifest_fields": list(MANIFEST_FIELDS),
    }
    meta_path = out_csv.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    sha_path = out_csv.with_suffix(".sha256")
    sha_path.write_text(digest + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d rows) sha256=%s", out_csv, len(rows), digest)
    return meta_out


def build_manifest(
    protocol: str,
    dataset_root: Path,
    categories: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    cats = list(categories or CATEGORIES)
    out = out_dir or (REPO_ROOT / "configs" / "splits")
    out.mkdir(parents=True, exist_ok=True)

    registry = load_protocol_registry()
    proto_meta = registry.get("protocols", {}).get(protocol, {})

    if protocol == "anomalib":
        rows = build_anomalib_protocol_rows(dataset_root, cats, seed=seed)
    elif protocol == "inferred":
        rows = build_inferred_protocol_rows(dataset_root, cats, seed=seed)
    elif protocol == "paper":
        rows = build_paper_protocol_rows(dataset_root, cats)
    else:
        raise ValueError(f"Unsupported protocol for manifest build: {protocol}")

    csv_path = out / f"{protocol}_manifest.csv"
    try:
        dataset_root_meta = str(dataset_root.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        dataset_root_meta = str(dataset_root.resolve()).replace("\\", "/")
    meta = {
        "protocol": protocol,
        "protocol_class": proto_meta.get("class"),
        "seed": seed,
        "dataset_root": dataset_root_meta,
        "categories": cats,
        "n_categories": len(cats),
        "status": proto_meta.get("status"),
        "notes": proto_meta.get("description"),
        "category_list_sha256": sha256_text(",".join(cats)),
    }
    return write_manifest(rows, csv_path, meta)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 3D-ADAM split manifests")
    p.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--protocol", choices=["anomalib", "inferred", "paper"], default="anomalib")
    p.add_argument("--categories", nargs="*")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "configs" / "splits")
    return p.parse_args()


def main() -> None:
    global logger
    args = parse_args()
    logger = setup_logger("build_split_manifest")
    try:
        meta = build_manifest(
            protocol=args.protocol,
            dataset_root=Path(args.root),
            categories=args.categories,
            seed=args.seed,
            out_dir=args.out_dir,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc
    print(json.dumps({"manifest_sha256": meta["manifest_sha256"], "n_rows": meta["n_rows"]}, indent=2))


if __name__ == "__main__":
    main()
