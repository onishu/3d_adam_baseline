"""Config parsing and runner dry-run / skip behavior."""

from __future__ import annotations

from pathlib import Path

from src.common.config import load_paper_metrics, load_protocol_registry, resolve_output_dir
from src.common.result_schema import ExperimentResult, ResultStatus
from src.common.runner import ExperimentRunner, RunnerConfig


def test_paper_metrics_table_separation() -> None:
    cfg = load_paper_metrics()
    assert "table4" in cfg and "table5" in cfg
    assert "patchcore_2d" in cfg["table4"]["models"]
    assert "patchcore" in cfg["table5"]["models"]
    # Table IV vs V PatchCore differ
    assert cfg["table4"]["models"]["patchcore_2d"]["image_AUROC"] != cfg["table5"]["models"]["patchcore"]["image_AUROC"]


def test_protocol_registry_and_output_dir() -> None:
    reg = load_protocol_registry()
    assert reg["protocols"]["paper"]["status"].startswith("blocked")
    out = resolve_output_dir("anomalib", None)
    assert "anomalib_protocol" in str(out).replace("\\", "/")


def test_runner_dry_run_and_skip(tmp_path: Path) -> None:
    cfg = RunnerConfig(
        model="dummy",
        table="table5",
        protocol="anomalib",
        output_dir=tmp_path,
        categories=["a", "b"],
        dry_run=True,
        force=False,
        resume=True,
    )
    runner = ExperimentRunner(cfg)

    def boom(_: str) -> ExperimentResult:
        raise AssertionError("should not run in dry-run")

    summary = runner.run(boom)
    assert summary["dry_run"] == ["a", "b"]
    assert (tmp_path / "raw" / "dummy" / "a.json").is_file()

    # resume should skip successful-looking dry_run? dry_run is not COMPLETED
    # so should_skip_existing returns False — re-write dry_run is ok
    summary2 = runner.run(boom)
    assert summary2["dry_run"] == ["a", "b"]

    # mark completed and ensure skip
    from src.common.result_schema import atomic_write_json

    path = tmp_path / "raw" / "dummy" / "a.json"
    atomic_write_json(
        path,
        ExperimentResult(
            experiment_id="anomalib:dummy:a",
            model="dummy",
            category="a",
            table="table5",
            protocol="anomalib",
            status=ResultStatus.COMPLETED.value,
            seed=42,
            image_AUROC=0.5,
        ).to_dict(),
    )
    summary3 = runner.run(boom)
    assert "a" in summary3["skipped"]
