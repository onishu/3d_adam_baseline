"""Tests for split manifest build/validate (CPU-only)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.data.build_split_manifest import (
    build_anomalib_protocol_rows,
    build_inferred_protocol_rows,
    build_manifest,
    holdout_indices,
    write_manifest,
)
from src.data.filename_parsing import parse_stem
from src.data.validate_split_manifest import validate_manifest


def test_parse_stem_patterns() -> None:
    p = parse_stem("000_A")
    assert p.part_id == "000"
    assert p.view_id == "A"
    assert p.parse_status == "parsed"

    n = parse_stem("012")
    assert n.part_id is None
    assert n.parse_status == "numeric_only"


def test_holdout_indices_deterministic() -> None:
    a = holdout_indices(48, 0.2, 42)
    b = holdout_indices(48, 0.2, 42)
    assert a == b
    train_idx, hold_idx = a
    assert len(train_idx) + len(hold_idx) == 48
    assert len(set(train_idx) & set(hold_idx)) == 0
    assert abs(len(hold_idx) / 48 - 0.2) < 0.05


def test_anomalib_manifest_on_tiny(tiny_adam3d: Path, tmp_path: Path) -> None:
    rows = build_anomalib_protocol_rows(tiny_adam3d, ["toy"], seed=42, test_split_ratio=0.2)
    splits = {r["split"] for r in rows}
    assert "train" in splits
    assert "test" in splits
    assert "validation" in splits
    assert "unused_on_disk_validation" in splits

    # SAME_AS_TEST: every test active sample should have a validation twin
    test_paths = {r["image_path"] for r in rows if r["split"] == "test"}
    val_paths = {r["image_path"] for r in rows if r["split"] == "validation"}
    assert test_paths == val_paths

    meta = write_manifest(
        rows,
        tmp_path / "anomalib_manifest.csv",
        {"protocol": "anomalib", "seed": 42},
    )
    assert meta["manifest_sha256"]
    # determinism
    meta2 = write_manifest(
        rows,
        tmp_path / "anomalib_manifest2.csv",
        {"protocol": "anomalib", "seed": 42},
    )
    assert meta["manifest_sha256"] == meta2["manifest_sha256"]

    report = validate_manifest(
        tmp_path / "anomalib_manifest.csv",
        tiny_adam3d,
        protocol="anomalib",
        strict_leakage=False,
        check_mask_content=True,
    )
    assert report["ok"], report["problems"]


def test_inferred_manifest_no_train_test_part_leak_when_parsed(tiny_adam3d: Path, tmp_path: Path) -> None:
    rows = build_inferred_protocol_rows(tiny_adam3d, ["toy"], seed=42)
    meta = write_manifest(rows, tmp_path / "inferred.csv", {"protocol": "inferred"})
    report = validate_manifest(
        tmp_path / "inferred.csv",
        tiny_adam3d,
        protocol="inferred",
        strict_leakage=True,
        check_mask_content=True,
    )
    assert report["ok"], report["problems"]
    critical = [x for x in report["part_id_leakage"] if x.get("scope") == "abnormal_only"]
    assert critical == []
    assert meta["n_rows"] > 0


def test_paper_protocol_blocked(tiny_adam3d: Path, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="blocked"):
        build_manifest(
            protocol="paper",
            dataset_root=tiny_adam3d,
            categories=["toy"],
            out_dir=tmp_path,
        )


@pytest.mark.dataset
def test_anomalib_manifest_matches_datamodule_counts() -> None:
    root = Path("datasets/ADAM3D")
    if not (root / "1m1").is_dir():
        pytest.skip("ADAM3D not available")

    from anomalib.data import ADAM3D

    rows = build_anomalib_protocol_rows(root, ["1m1"], seed=42)
    active = [r for r in rows if r["split"] in {"train", "validation", "test"}]
    # unique paths per split (validation duplicates test paths)
    train_n = len({r["image_path"] for r in active if r["split"] == "train"})
    test_n = len({r["image_path"] for r in active if r["split"] == "test"})

    dm = ADAM3D(root=str(root), category="1m1", num_workers=0, seed=42)
    dm.prepare_data()
    dm.setup()
    assert train_n == len(dm.train_data)
    assert test_n == len(dm.test_data)
    assert len(dm.val_data) == len(dm.test_data)


@pytest.mark.dataset
def test_real_manifest_csv_schema_roundtrip(tmp_path: Path) -> None:
    root = Path("datasets/ADAM3D")
    if not (root / "1m1").is_dir():
        pytest.skip("ADAM3D not available")
    meta = build_manifest(
        protocol="anomalib",
        dataset_root=root,
        categories=["1m1"],
        out_dir=tmp_path,
    )
    path = Path(meta["manifest_csv"])
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        assert "part_id" in reader.fieldnames
        assert "split" in reader.fieldnames
        row = next(reader)
        assert row["category"] == "1m1"
