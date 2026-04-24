"""Fast unit tests for paper_embedder.markdown.

Mocks Marker internals (_run_marker, _find_section_pages) so no real Marker
model load or PDF parsing happens. Exercises cache semantics, oversize branch,
post-trim logic, compress-fallback, and the eager import guard.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from paper_embedder.markdown import (
    LARGE_PDF_THRESHOLD_BYTES,
    MIN_OUTPUT_CHARS,
    STOP_KEYWORDS,
    _compress_pdf,
    _extract_sections,
    prepare_fulltext,
)


# ---------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------


def test_public_symbol_callable():
    assert callable(prepare_fulltext)


def test_stop_keywords_includes_expected_terminators():
    # Sanity: at least these must be present (see hardening spec §4).
    for kw in ("experiment", "result", "conclusion", "discussion"):
        assert kw in STOP_KEYWORDS


def test_constants_are_sensible():
    assert LARGE_PDF_THRESHOLD_BYTES == 50 * 1024 * 1024
    assert MIN_OUTPUT_CHARS == 100


# ---------------------------------------------------------------
# _extract_sections (pure string function)
# ---------------------------------------------------------------


def test_extract_sections_trims_at_first_stop_keyword():
    md = (
        "# Title\n\n"
        "## Abstract\n\nabstract text\n\n"
        "## 1. Introduction\n\nintro text\n\n"
        "## 2. Method\n\nmethod text\n\n"
        "## 3. Experiments\n\nexperiments text\n\n"
        "## 4. Conclusion\n\nconclusion text\n"
    )
    result = _extract_sections(md)
    assert "1. Introduction" in result
    assert "intro text" in result
    assert "2. Method" in result
    assert "method text" in result
    assert "3. Experiments" not in result
    assert "experiments text" not in result


def test_extract_sections_applications_is_a_stop_keyword():
    md = (
        "## 1. Introduction\n\nintro\n\n"
        "## 2. Method\n\nmethod\n\n"
        "## 3. Applications\n\napps\n"
    )
    result = _extract_sections(md)
    assert "1. Introduction" in result
    assert "3. Applications" not in result


def test_extract_sections_returns_input_when_no_headings():
    md = "just some markdown text without any numbered headings\n"
    result = _extract_sections(md)
    assert result == md


# ---------------------------------------------------------------
# _compress_pdf (ghostscript wrapper)
# ---------------------------------------------------------------


def test_compress_pdf_returns_false_on_missing_source(tmp_path):
    src = tmp_path / "missing.pdf"
    dst = tmp_path / "out.pdf"

    ok = _compress_pdf(src, dst)

    assert ok is False
    assert not dst.exists()


# ---------------------------------------------------------------
# prepare_fulltext — pipeline + cache semantics (all mocked)
# ---------------------------------------------------------------

_FAKE_MD = (
    "## 1. Introduction\n\nHello intro. This paper presents a novel approach.\n\n"
    "## 2. Method\n\nHello method. We describe our methodology in detail here.\n\n"
    "## 3. Experiments\n\nHello experiments.\n"
)
_FAKE_PAGE_RANGE = ([0, 1, 2, 3, 4, 5], {"method": "fallback", "total_pages": 6})


def _make_small_pdf(tmp_path: Path) -> Path:
    """Create a small dummy file that prepare_fulltext treats as a PDF.

    We mock _find_section_pages so it never actually parses the file — only
    .exists() and .stat().st_size are consulted by prepare_fulltext itself.
    """
    p = tmp_path / "paper.pdf"
    p.write_bytes(b"%PDF-1.4\n%dummy\n")
    return p


def _make_large_pdf(tmp_path: Path) -> Path:
    """Create a >50 MB dummy file to trigger the compression branch."""
    p = tmp_path / "big.pdf"
    with p.open("wb") as fh:
        fh.write(b"%PDF-1.4\n")
        fh.write(b"\x00" * (LARGE_PDF_THRESHOLD_BYTES + 1024))
    return p


def test_prepare_fulltext_returns_trimmed_markdown(tmp_path):
    pdf = _make_small_pdf(tmp_path)
    cache = tmp_path / "cache.md"
    with patch("paper_embedder.markdown._find_section_pages", return_value=_FAKE_PAGE_RANGE), \
         patch("paper_embedder.markdown._run_marker", return_value=_FAKE_MD):
        out = prepare_fulltext(pdf, cache_path=cache)

    assert out is not None
    assert "1. Introduction" in out
    assert "2. Method" in out
    assert "3. Experiments" not in out
    assert cache.exists()
    assert cache.read_text(encoding="utf-8") == out


def test_prepare_fulltext_cache_hit_skips_pipeline(tmp_path):
    pdf = _make_small_pdf(tmp_path)
    cache = tmp_path / "cache.md"
    cache.write_text("# CACHED CONTENT\n\nline 1\n" + "x" * 200, encoding="utf-8")

    with patch("paper_embedder.markdown._find_section_pages") as mock_scan, \
         patch("paper_embedder.markdown._run_marker") as mock_marker:
        out = prepare_fulltext(pdf, cache_path=cache)

    mock_scan.assert_not_called()
    mock_marker.assert_not_called()
    assert out is not None
    assert "CACHED CONTENT" in out


def test_prepare_fulltext_no_cache_path_runs_pipeline(tmp_path):
    pdf = _make_small_pdf(tmp_path)
    with patch("paper_embedder.markdown._find_section_pages", return_value=_FAKE_PAGE_RANGE), \
         patch("paper_embedder.markdown._run_marker", return_value=_FAKE_MD):
        out = prepare_fulltext(pdf, cache_path=None)

    assert out is not None
    assert "1. Introduction" in out


def test_prepare_fulltext_returns_none_when_marker_raises(tmp_path):
    pdf = _make_small_pdf(tmp_path)
    cache = tmp_path / "cache.md"
    with patch("paper_embedder.markdown._find_section_pages", return_value=_FAKE_PAGE_RANGE), \
         patch("paper_embedder.markdown._run_marker", side_effect=RuntimeError("boom")):
        out = prepare_fulltext(pdf, cache_path=cache)

    assert out is None
    assert not cache.exists()


def test_prepare_fulltext_returns_none_when_marker_returns_empty(tmp_path):
    pdf = _make_small_pdf(tmp_path)
    with patch("paper_embedder.markdown._find_section_pages", return_value=_FAKE_PAGE_RANGE), \
         patch("paper_embedder.markdown._run_marker", return_value=""):
        out = prepare_fulltext(pdf)

    assert out is None


def test_prepare_fulltext_returns_none_when_output_too_short(tmp_path):
    pdf = _make_small_pdf(tmp_path)
    tiny = "## 1. Introduction\nhi\n## 2. Results\n..."  # trimmed < 100 chars
    with patch("paper_embedder.markdown._find_section_pages", return_value=_FAKE_PAGE_RANGE), \
         patch("paper_embedder.markdown._run_marker", return_value=tiny):
        out = prepare_fulltext(pdf)

    assert out is None


def test_prepare_fulltext_treats_corrupt_cache_as_miss(tmp_path):
    pdf = _make_small_pdf(tmp_path)
    cache = tmp_path / "cache.md"
    cache.write_bytes(b"\xff\xfe\x00\x01garbage")  # invalid UTF-8

    with patch("paper_embedder.markdown._find_section_pages", return_value=_FAKE_PAGE_RANGE), \
         patch("paper_embedder.markdown._run_marker", return_value=_FAKE_MD):
        out = prepare_fulltext(pdf, cache_path=cache)

    assert out is not None
    assert "1. Introduction" in out
    assert cache.read_text(encoding="utf-8") == out


def test_prepare_fulltext_treats_empty_cache_as_miss(tmp_path):
    pdf = _make_small_pdf(tmp_path)
    cache = tmp_path / "cache.md"
    cache.write_text("", encoding="utf-8")

    with patch("paper_embedder.markdown._find_section_pages", return_value=_FAKE_PAGE_RANGE), \
         patch("paper_embedder.markdown._run_marker", return_value=_FAKE_MD):
        out = prepare_fulltext(pdf, cache_path=cache)

    assert out is not None
    assert "1. Introduction" in out


def test_prepare_fulltext_oversize_triggers_compression(tmp_path):
    pdf = _make_large_pdf(tmp_path)
    with patch("paper_embedder.markdown._find_section_pages", return_value=_FAKE_PAGE_RANGE), \
         patch("paper_embedder.markdown._compress_pdf", return_value=True) as mock_compress, \
         patch("paper_embedder.markdown._run_marker", return_value=_FAKE_MD):
        out = prepare_fulltext(pdf)

    mock_compress.assert_called_once()
    assert out is not None
    assert "1. Introduction" in out


def test_prepare_fulltext_oversize_compression_failure_returns_none(tmp_path):
    pdf = _make_large_pdf(tmp_path)
    with patch("paper_embedder.markdown._find_section_pages", return_value=_FAKE_PAGE_RANGE), \
         patch("paper_embedder.markdown._compress_pdf", return_value=False), \
         patch("paper_embedder.markdown._run_marker") as mock_marker:
        out = prepare_fulltext(pdf)

    assert out is None
    mock_marker.assert_not_called()


# ---------------------------------------------------------------
# Eager import guard
# ---------------------------------------------------------------


def test_import_guard_raises_when_marker_missing(monkeypatch):
    """Simulate marker-pdf not installed by shadowing `marker` in sys.modules
    and forcing a reimport of paper_embedder.markdown."""
    import importlib
    import sys

    # Remove the cached module and hide marker from future imports.
    sys.modules.pop("paper_embedder.markdown", None)
    monkeypatch.setitem(sys.modules, "marker", None)  # None triggers ImportError on import

    with pytest.raises(ImportError, match=r"\[marker\] extra"):
        importlib.import_module("paper_embedder.markdown")

    # Cleanup: remove the None shadow and reimport so subsequent tests see the real module.
    sys.modules.pop("marker", None)
    sys.modules.pop("paper_embedder.markdown", None)
    importlib.import_module("paper_embedder.markdown")
