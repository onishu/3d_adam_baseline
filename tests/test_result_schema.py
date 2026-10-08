"""Result JSON schema, atomic write, skip/retry behavior."""

from __future__ import annotations

import json
from pathlib import Path

from src.common.result_schema import (
    ExperimentResult,
    ResultStatus,
    atomic_write_json,
    load_result_json,
    should_skip_existing,
    validate_result_dict,
)


def test_validate_result_dict_required_keys() -> None:
    bad = {"model": "x"}
    problems = validate_result_dict(bad)
    assert any(p.startswith("missing_or_empty:experiment_id") for p in problems)


def test_atomic_write_and_load(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "toy.json"
    result = ExperimentResult(
        experiment_id="anomalib:patchcore:1m1",
        model="patchcore",
        category="1m1",
        table="table5",
        protocol="anomalib",
        status=ResultStatus.COMPLETED.value,
        seed=42,
        image_AUROC=0.9,
    )
    atomic_write_json(path, result.to_dict())
    assert path.is_file()
    assert not list(tmp_path.rglob("*.tmp"))
    loaded = load_result_json(path)
    assert loaded.image_AUROC == 0.9
    assert loaded.is_successful()


def test_skip_completed_retry_failed(tmp_path: Path) -> None:
    ok = tmp_path / "ok.json"
    bad = tmp_path / "bad.json"
    atomic_write_json(
        ok,
        ExperimentResult(
            experiment_id="e1",
            model="m",
            category="c",
            table="table5",
            protocol="anomalib",
            status=ResultStatus.COMPLETED.value,
        ).to_dict(),
    )
    atomic_write_json(
        bad,
        ExperimentResult(
            experiment_id="e2",
            model="m",
            category="c2",
            table="table5",
            protocol="anomalib",
            status=ResultStatus.FAILED.value,
            error_type="RuntimeError",
            error_message="boom",
        ).to_dict(),
    )
    assert should_skip_existing(ok, force=False, resume=True) is True
    assert should_skip_existing(ok, force=True, resume=True) is False
    assert should_skip_existing(bad, force=False, resume=True) is False


def test_windows_path_in_json(tmp_path: Path) -> None:
    path = tmp_path / "win.json"
    payload = {
        "experiment_id": "e",
        "model": "m",
        "category": "c",
        "table": "table5",
        "protocol": "anomalib",
        "status": "completed",
        "seed": 42,
        "dataset_root": str(tmp_path / "datasets" / "ADAM3D"),
        "traceback_path": str(tmp_path / "logs" / "c.traceback.txt"),
    }
    atomic_write_json(path, payload)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "ADAM3D" in data["dataset_root"]
