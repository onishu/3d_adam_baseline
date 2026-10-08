"""Logging helpers with Windows-friendly defaults."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str,
    log_file: Path | str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger that writes to stderr and optionally a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
