"""Section-aware PDF extractor — extracts 'Introduction + Methods' region.

Ported from My-Research-Assistant's converters/section_extract.py with a cleaner
public API. The heuristic: find numbered section headings on each page, stop when
a heading's title matches a known terminator (Results/Experiments/etc.), return
the text between page 0 and that stop page.
"""

from __future__ import annotations

import re


_HEADING_RE = re.compile(
    r"^\s*(\d+)\.?\s+([A-Z][A-Za-z][A-Za-z ]{2,40})\s*$"
)

_STOP_KEYWORDS = frozenset({
    "results",
    "experiments",
    "experiment",
    "conclusion",
    "conclusions",
    "discussion",
    "references",
    "acknowledgments",
    "acknowledgements",
    "bibliography",
    "appendix",
})


def _find_heading_on_line(line: str) -> tuple[int, str] | None:
    """Return (section_number, title) if this line is a numbered section heading."""
    m = _HEADING_RE.match(line)
    if not m:
        return None
    number, title = int(m.group(1)), m.group(2).strip()
    return number, title


def _is_stop_keyword(title: str) -> bool:
    """Case-insensitive check for section-heading terminators."""
    return title.strip().lower() in _STOP_KEYWORDS
