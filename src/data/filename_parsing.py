"""Parse ADAM3D anomalib-subset filenames into part/view identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Good samples typically: 000_A.png, 015_D.png
# Some defect folders keep the same pattern; others are renamed to 000.png
_PART_VIEW_RE = re.compile(r"^(?P<part>\d+)_(?P<view>[A-Za-z0-9]+)$")
_NUMERIC_RE = re.compile(r"^(?P<part>\d+)$")


@dataclass(frozen=True)
class ParsedStem:
    stem: str
    part_id: str | None
    view_id: str | None
    parse_status: str  # parsed | numeric_only | unknown
    confidence: str  # high | low | none


def parse_stem(stem: str) -> ParsedStem:
    """Parse a file stem into part_id / view_id when possible.

    Notes
    -----
    - ``000_A`` is treated as part_id=000, view_id=A (assumption: letter = view).
    - ``000`` alone is unparseable for part-instance leakage checks
      (anomalib subset often renames defects this way).
    """
    m = _PART_VIEW_RE.match(stem)
    if m:
        return ParsedStem(
            stem=stem,
            part_id=m.group("part"),
            view_id=m.group("view"),
            parse_status="parsed",
            confidence="high",
        )
    m2 = _NUMERIC_RE.match(stem)
    if m2:
        return ParsedStem(
            stem=stem,
            part_id=None,
            view_id=None,
            parse_status="numeric_only",
            confidence="none",
        )
    return ParsedStem(
        stem=stem,
        part_id=None,
        view_id=None,
        parse_status="unknown",
        confidence="none",
    )
