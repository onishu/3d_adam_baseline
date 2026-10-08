"""Config loading and shared CLI option definitions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "ADAM3D"
DEFAULT_SEED = 42


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {p}, got {type(data).__name__}")
    return data


def load_paper_metrics(path: Path | str | None = None) -> dict[str, Any]:
    """Load centralized paper metric targets."""
    cfg = Path(path) if path else REPO_ROOT / "configs" / "paper_metrics.yaml"
    return load_yaml(cfg)


def load_protocol_registry(path: Path | str | None = None) -> dict[str, Any]:
    """Load split protocol registry."""
    cfg = Path(path) if path else REPO_ROOT / "configs" / "splits" / "protocols.yaml"
    return load_yaml(cfg)


def add_common_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach CLI flags shared by benchmark entrypoints."""
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--protocol",
        default="anomalib_subset",
        choices=[
            "anomalib",
            "anomalib_subset",
            "anomalib_same_as_test",
            "disk_validation",
            "inferred_part_split",
            "paper",
            "inferred",
            "author_cfa",
        ],
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    return parser


def resolve_output_dir(protocol: str, output_dir: Path | None) -> Path:
    """Resolve result root for a protocol without overwriting legacy trees by default."""
    if output_dir is not None:
        return Path(output_dir)
    key = "anomalib" if protocol == "anomalib_subset" else protocol
    registry = load_protocol_registry()
    proto = registry.get("protocols", {}).get(key, {})
    sub = proto.get("result_subdir")
    if sub:
        return REPO_ROOT / Path(sub)
    return REPO_ROOT / "results" / protocol
