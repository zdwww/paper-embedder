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


from pathlib import Path


# Indirection so tests can monkeypatch without importing pdfminer.
def _pdfminer_extract_pages(path: Path):
    from pdfminer.high_level import extract_pages
    return extract_pages(str(path))


def _pdfminer_extract_text(path: Path, page_numbers: list[int]) -> str:
    from pdfminer.high_level import extract_text
    return extract_text(str(path), page_numbers=page_numbers)


def extract_intro_and_methods(pdf_path: Path, *, max_scan_pages: int = 10) -> str | None:
    """Extract 'Introduction + Methods' text from a PDF.

    Returns concatenated page-text from page 0 up to (exclusive) the first page
    whose numbered heading matches a STOP keyword (Results/Experiments/Conclusion/...).
    Returns None if no headings are found, or if any error occurs (missing file,
    corrupted PDF, empty document).

    Never raises. Caller should treat None as 'no fulltext vector available'.
    """
    path = Path(pdf_path)
    if not path.exists() or not path.is_file():
        return None

    try:
        # Enumerate pages first so we know how many we have.
        try:
            page_count = sum(1 for _ in _pdfminer_extract_pages(path))
        except Exception:
            return None

        if page_count == 0:
            return None

        pages_to_scan = min(page_count, max_scan_pages)
        found_any_heading = False
        accumulated: list[str] = []

        for i in range(pages_to_scan):
            try:
                page_text = _pdfminer_extract_text(path, page_numbers=[i])
            except Exception:
                return None

            # Check this page for a STOP heading BEFORE appending everything.
            truncate_at: int | None = None
            for line in page_text.splitlines():
                heading = _find_heading_on_line(line)
                if heading is None:
                    continue
                found_any_heading = True
                _, title = heading
                if _is_stop_keyword(title):
                    truncate_at = page_text.find(line)
                    break

            if truncate_at is not None:
                accumulated.append(page_text[:truncate_at])
                break

            accumulated.append(page_text)

        if not found_any_heading:
            return None

        return "".join(accumulated).strip() or None

    except Exception:
        return None
