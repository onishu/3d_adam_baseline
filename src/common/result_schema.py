"""Typed experiment result schema and atomic JSON I/O."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class ResultStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    DRY_RUN = "dry_run"
    SMOKE = "smoke"


REQUIRED_RESULT_KEYS = (
    "experiment_id",
    "model",
    "category",
    "table",
    "protocol",
    "status",
    "seed",
)


@dataclass
class ExperimentResult:
    """Common per-category experiment record."""

    experiment_id: str
    model: str
    category: str
    table: str
    protocol: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    seed: int = 42
    dataset_root: str | None = None
    dataset_hash: str | None = None
    split_manifest_sha256: str | None = None
    git_commit: str | None = None
    dirty_worktree: bool | None = None
    python_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_vram: int | None = None
    package_versions: dict[str, Any] = field(default_factory=dict)
    external_repo_url: str | None = None
    external_repo_commit: str | None = None
    pretrained_weight_path: str | None = None
    pretrained_weight_sha256: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    data_config: dict[str, Any] = field(default_factory=dict)
    metric_config: dict[str, Any] = field(default_factory=dict)
    image_AUROC: float | None = None
    image_F1Score: float | None = None
    pixel_AUROC: float | None = None
    pixel_F1Score: float | None = None
    AUPRO: float | None = None
    # Oracle / test-optimized F1 (explicitly separated from validation-threshold F1)
    image_F1Score_test_optimized: float | None = None
    pixel_F1Score_test_optimized: float | None = None
    peak_gpu_memory_mb: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback_path: str | None = None
    notes: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        extras = {k: v for k, v in data.items() if k not in known}
        if extras:
            merged = dict(kwargs.get("extras") or {})
            merged.update(extras)
            kwargs["extras"] = merged
        return cls(**kwargs)

    def is_successful(self) -> bool:
        return self.status == ResultStatus.COMPLETED.value and self.error_type is None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def validate_result_dict(data: dict[str, Any]) -> list[str]:
    """Return list of schema problems (empty if OK)."""
    problems: list[str] = []
    for key in REQUIRED_RESULT_KEYS:
        if key not in data or data[key] in (None, ""):
            problems.append(f"missing_or_empty:{key}")
    status = data.get("status")
    if status is not None and status not in {s.value for s in ResultStatus}:
        problems.append(f"invalid_status:{status}")
    return problems


def atomic_write_json(path: Path | str, data: dict[str, Any]) -> None:
    """Write JSON atomically (temp file + replace). Cleans up temp on failure."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def load_result_json(path: Path | str) -> ExperimentResult:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Result JSON must be an object: {p}")
    problems = validate_result_dict(data)
    if problems:
        raise ValueError(f"Invalid result schema in {p}: {problems}")
    return ExperimentResult.from_dict(data)


def should_skip_existing(path: Path, force: bool, resume: bool) -> bool:
    """Skip completed results unless --force. Failed results are retryable."""
    if force or not path.exists():
        return False
    if not resume:
        return False
    try:
        result = load_result_json(path)
    except Exception:
        return False
    return result.is_successful()
