"""CLI to capture environment JSON for reproducibility audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .environment import capture_environment
from .hashing import sha256_file
from .result_schema import atomic_write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture reproduction environment metadata")
    p.add_argument("--out", type=Path, default=Path("results/summary/environment.json"))
    p.add_argument("--include-freeze", action="store_true")
    p.add_argument("--manifest", type=Path, default=None, help="Optional split manifest for SHA-256")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = capture_environment(include_freeze=args.include_freeze)
    if args.manifest is not None:
        if not args.manifest.is_file():
            raise SystemExit(f"Manifest not found: {args.manifest}")
        info["dataset_manifest_sha256"] = sha256_file(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, info)
    print(json.dumps({"wrote": str(args.out), "git_commit": info.get("git_commit")}, indent=2))


if __name__ == "__main__":
    main()
