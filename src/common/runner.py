"""Category-loop experiment runner with skip/resume and atomic writes."""

from __future__ import annotations

import gc
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .environment import capture_environment
from .logging_utils import setup_logger
from .result_schema import (
    ExperimentResult,
    ResultStatus,
    atomic_write_json,
    should_skip_existing,
    utc_now_iso,
)

CategoryRunner = Callable[[str], ExperimentResult]


@dataclass
class RunnerConfig:
    model: str
    table: str
    protocol: str
    output_dir: Path
    categories: Sequence[str]
    seed: int = 42
    force: bool = False
    resume: bool = True
    dry_run: bool = False
    smoke: bool = False
    dataset_root: str | None = None
    split_manifest_sha256: str | None = None


class ExperimentRunner:
    """Run per-category experiments with atomic result persistence."""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.raw_dir = Path(config.output_dir) / "raw" / config.model
        self.log_dir = Path(config.output_dir) / "logs" / config.model
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            f"runner.{config.model}",
            self.log_dir / "runner.log",
        )
        self.env = capture_environment()

    def result_path(self, category: str) -> Path:
        return self.raw_dir / f"{category}.json"

    def _base_result(self, category: str, status: str) -> ExperimentResult:
        env = self.env
        return ExperimentResult(
            experiment_id=f"{self.config.protocol}:{self.config.model}:{category}",
            model=self.config.model,
            category=category,
            table=self.config.table,
            protocol=self.config.protocol,
            status=status,
            seed=self.config.seed,
            dataset_root=self.config.dataset_root,
            split_manifest_sha256=self.config.split_manifest_sha256,
            git_commit=env.get("git_commit"),
            dirty_worktree=env.get("dirty_worktree"),
            python_version=env.get("python_version"),
            torch_version=env.get("torch_version") or env.get("package_versions", {}).get("torch"),
            cuda_version=env.get("cuda_version"),
            gpu_name=env.get("gpu_name"),
            gpu_vram=env.get("gpu_vram"),
            package_versions=env.get("package_versions") or {},
        )

    def run(self, run_category: CategoryRunner) -> dict[str, Any]:
        summary = {"completed": [], "skipped": [], "failed": [], "dry_run": [], "smoke": []}
        categories = list(self.config.categories)
        if self.config.smoke and categories:
            categories = categories[:1]

        for index, category in enumerate(categories, 1):
            path = self.result_path(category)
            prefix = f"[{index}/{len(categories)}]"
            if should_skip_existing(path, self.config.force, self.config.resume):
                self.logger.info("%s SKIP %s/%s", prefix, self.config.model, category)
                summary["skipped"].append(category)
                continue

            if self.config.dry_run:
                result = self._base_result(category, ResultStatus.DRY_RUN.value)
                result.started_at = utc_now_iso()
                result.finished_at = utc_now_iso()
                result.notes = "dry-run only; no training executed"
                atomic_write_json(path, result.to_dict())
                self.logger.info("%s DRY-RUN %s/%s", prefix, self.config.model, category)
                summary["dry_run"].append(category)
                continue

            self.logger.info("%s RUN %s/%s", prefix, self.config.model, category)
            tb_path = self.log_dir / f"{category}.traceback.txt"
            started = utc_now_iso()
            try:
                result = run_category(category)
                if result.started_at is None:
                    result.started_at = started
                if result.finished_at is None:
                    result.finished_at = utc_now_iso()
                if self.config.smoke and result.status == ResultStatus.COMPLETED.value:
                    result.status = ResultStatus.SMOKE.value
                # Fill env defaults if omitted by caller.
                self._fill_defaults(result)
                atomic_write_json(path, result.to_dict())
                bucket = "smoke" if result.status == ResultStatus.SMOKE.value else "completed"
                if result.status == ResultStatus.FAILED.value:
                    bucket = "failed"
                summary[bucket].append(category)
            except Exception as exc:
                tb = traceback.format_exc()
                tb_path.write_text(tb, encoding="utf-8")
                result = self._base_result(category, ResultStatus.FAILED.value)
                result.started_at = started
                result.finished_at = utc_now_iso()
                result.error_type = type(exc).__name__
                result.error_message = str(exc)
                result.traceback_path = str(tb_path)
                atomic_write_json(path, result.to_dict())
                self.logger.error("%s ERROR %s: %s", prefix, type(exc).__name__, exc)
                summary["failed"].append(category)
            finally:
                self._release_gpu()

        self.logger.info("Done: %s", self.raw_dir)
        return summary

    def _fill_defaults(self, result: ExperimentResult) -> None:
        if not result.dataset_root:
            result.dataset_root = self.config.dataset_root
        if not result.split_manifest_sha256:
            result.split_manifest_sha256 = self.config.split_manifest_sha256
        if result.git_commit is None:
            result.git_commit = self.env.get("git_commit")
        if result.dirty_worktree is None:
            result.dirty_worktree = self.env.get("dirty_worktree")
        if not result.package_versions:
            result.package_versions = self.env.get("package_versions") or {}

    @staticmethod
    def _release_gpu() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
