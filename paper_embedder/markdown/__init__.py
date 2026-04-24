"""Hardened Marker extraction wrapper.

Design: docs/superpowers/specs/2026-04-23-marker-hardening-design.md
Validated by the M4 V2 benchmark (10/10 on the Plan D corpus, 2026-04-23).

Public API:
    prepare_fulltext(pdf_path, *, cache_path=None) -> str | None

Rules:
    R1 - fresh PdfConverter per paper (never reuse)
    R2 - always pre-scan + page_range + post-trim (MRA pattern)
"""

from __future__ import annotations

try:
    import marker  # noqa: F401
except ImportError as e:
    raise ImportError(
        "paper_embedder.markdown requires the [marker] extra. "
        "Install with: pip install 'paper-embedder[marker]'"
    ) from e

import logging
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

LARGE_PDF_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB
FALLBACK_MAX_PAGE = 5  # matches MRA/converters/section_extract.py
MIN_OUTPUT_CHARS = 100  # below this, treat extraction as failure

STOP_KEYWORDS = [
    "experiment", "result", "evaluation", "empirical", "ablation",
    "conclusion", "discussion", "limitation", "summary", "future work",
    "application", "deployment",
]

# Matches top-level numbered section headings in raw PDF text.
# e.g. "1. Introduction", "4. Experiments and Results", "3 Method"
_HEADING_RE = re.compile(
    r'^\s*(\d+)[. ][ \t]*([A-Z][^\r\n]{2,})',
    re.MULTILINE,
)

# Matches markdown headings like "# 1. Introduction", "## 3. Method", etc.
_MD_HEADING_RE = re.compile(
    r'^(#{1,4})\s+(\d+)[.\s]\s+(.+)$',
    re.MULTILINE,
)

_MODELS: object | None = None


def _get_models() -> object:
    """Load Marker's model dict once per process; cached."""
    global _MODELS
    if _MODELS is None:
        from marker.models import create_model_dict
        _MODELS = create_model_dict()
    return _MODELS


def _find_section_pages(pdf_path: Path) -> tuple[list[int], dict[str, object]]:
    """Phase 1 - pypdfium2 pre-scan. Returns (page_range, metadata).

    Port of MRA/converters/section_extract.py lines 71-143.
    """
    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    doc = pdfium.PdfDocument(str(pdf_path))
    page_count = len(doc)

    sections: dict[int, tuple[int, str]] = {}  # sec_num -> (page_idx, title)
    scan_limit = min(10, page_count)
    for page_idx in range(scan_limit):
        page = doc[page_idx]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded()
        textpage.close()
        page.close()

        for m in _HEADING_RE.finditer(text):
            sec_num = int(m.group(1))
            title = m.group(2).strip().rstrip('.')
            if len(title) < 3:
                continue
            if sec_num not in sections:
                sections[sec_num] = (page_idx, title)

    doc.close()

    stop_page = None
    stop_section = None
    method = "fallback"
    for sec_num in sorted(sections.keys()):
        page_idx, title = sections[sec_num]
        title_lower = title.lower()
        if any(kw in title_lower for kw in STOP_KEYWORDS):
            stop_page = page_idx
            stop_section = f"{sec_num}. {title}"
            method = "stop_keyword"
            break

    if stop_page is None:
        stop_page = min(FALLBACK_MAX_PAGE, page_count - 1)

    page_range = list(range(0, stop_page + 1))
    if len(page_range) < 2:
        page_range = list(range(0, min(FALLBACK_MAX_PAGE + 1, page_count)))

    metadata: dict[str, object] = {
        "total_pages": page_count,
        "pages_converted": len(page_range),
        "stop_page": stop_page,
        "stop_section": stop_section,
        "method": method,
        "sections_found": {
            str(k): {"page": v[0], "title": v[1]} for k, v in sections.items()
        },
    }
    return page_range, metadata


def _compress_pdf(src: Path, dst: Path) -> bool:
    """Phase 1.5 - ghostscript /ebook compression. Returns True on success.

    Downsamples embedded images to 150 DPI. Text and vector content
    untouched. Returns False (no exception) on any failure: missing source,
    ghostscript binary unavailable, nonzero exit, empty output.
    """
    if not src.exists() or not src.is_file():
        logger.warning("_compress_pdf: source missing: %s", src)
        return False

    cmd = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/ebook",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={dst}",
        str(src),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        logger.warning("_compress_pdf: ghostscript 'gs' not installed")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("_compress_pdf: timeout compressing %s", src)
        return False

    if result.returncode != 0:
        logger.warning(
            "_compress_pdf: gs exit %d on %s; stderr=%s",
            result.returncode, src, result.stderr[:200],
        )
        return False
    if not dst.exists() or dst.stat().st_size == 0:
        logger.warning("_compress_pdf: empty output for %s", src)
        return False
    return True


def _run_marker(pdf_path: Path, page_range: list[int]) -> str:
    """Phase 2 - fresh PdfConverter with page_range config. Returns markdown.

    Rule R1: fresh PdfConverter per call. Never reused.
    Rule R2: always honor page_range from pre-scan.
    """
    from marker.converters.pdf import PdfConverter
    from marker.output import text_from_rendered

    conv = PdfConverter(
        artifact_dict=_get_models(),
        config={"page_range": page_range},
    )
    try:
        rendered = conv(str(pdf_path))
        text, _, _ = text_from_rendered(rendered)
        return str(text)  # text_from_rendered returns Any; cast to str
    finally:
        del conv


def _extract_sections(markdown_text: str) -> str:
    """Phase 3 - trim markdown from section 1 to first stop heading.

    Port of MRA/converters/section_extract.py lines 160-211 (image handling
    stripped — the wrapper does not persist images).
    """
    headings: list[dict[str, int | str]] = []
    for m in _MD_HEADING_RE.finditer(markdown_text):
        headings.append({
            "pos": m.start(),
            "sec_num": int(m.group(2)),
            "title": m.group(3).strip(),
        })

    if not headings:
        return markdown_text

    intro_start: int | None = None
    for h in headings:
        if h["sec_num"] == 1:
            intro_start = int(h["pos"])
            break
    if intro_start is None:
        intro_start = 0

    stop_pos: int | None = None
    for h in headings:
        title_lower = str(h["title"]).lower()
        if any(kw in title_lower for kw in STOP_KEYWORDS):
            stop_pos = int(h["pos"])
            break

    if stop_pos is None:
        return markdown_text[intro_start:]
    if stop_pos <= intro_start:
        return markdown_text[intro_start:]
    return markdown_text[intro_start:stop_pos].rstrip()


def prepare_fulltext(
    pdf_path: Path,
    *,
    cache_path: Path | None = None,
) -> str | None:
    """Extract intro+methods markdown from a PDF.

    Cache semantics (when cache_path is provided):
      - cache_path exists, valid UTF-8, non-empty -> return its contents.
      - cache_path exists but unreadable/empty/invalid -> treat as miss,
        re-extract, overwrite.
      - cache_path does not exist -> run pipeline, write cache on success.

    If cache_path is None, always run the pipeline; do not persist.

    Returns None on any failure. Never raises.
    """
    pdf_path = Path(pdf_path)

    # Cache read
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                cached = cache_path.read_text(encoding="utf-8")
                if cached:
                    return cached
                logger.warning("prepare_fulltext: cache empty at %s; re-extracting", cache_path)
            except (UnicodeDecodeError, OSError) as e:
                logger.warning(
                    "prepare_fulltext: cache unreadable at %s (%s); re-extracting",
                    cache_path, e,
                )

    # Pipeline
    tmp_compressed: Path | None = None
    try:
        # Phase 1 - pre-scan
        page_range, _ = _find_section_pages(pdf_path)

        # Phase 1.5 - optional compression for oversized inputs
        marker_input = pdf_path
        if pdf_path.stat().st_size > LARGE_PDF_THRESHOLD_BYTES:
            fd = tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False, dir=tempfile.gettempdir(),
            )
            fd.close()
            tmp_compressed = Path(fd.name)
            if not _compress_pdf(pdf_path, tmp_compressed):
                logger.warning(
                    "prepare_fulltext: compression failed for oversize PDF %s",
                    pdf_path,
                )
                return None
            marker_input = tmp_compressed

        # Phase 2 - Marker with fresh converter + page_range
        raw_md = _run_marker(marker_input, page_range)
        if not raw_md:
            logger.warning("prepare_fulltext: Marker returned empty text for %s", pdf_path)
            return None

        # Phase 3 - post-trim
        trimmed = _extract_sections(raw_md)
        if not trimmed or len(trimmed) < MIN_OUTPUT_CHARS:
            logger.warning(
                "prepare_fulltext: trimmed output too short (%d chars) for %s",
                len(trimmed) if trimmed else 0, pdf_path,
            )
            return None

        # Cache write
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(trimmed, encoding="utf-8")
            except OSError as e:
                logger.warning(
                    "prepare_fulltext: cache write failed for %s (%s); returning text anyway",
                    cache_path, e,
                )

        return trimmed

    except Exception as e:
        logger.warning(
            "prepare_fulltext failed for %s: %s", pdf_path, e, exc_info=True,
        )
        return None

    finally:
        if tmp_compressed is not None:
            try:
                tmp_compressed.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["prepare_fulltext"]
