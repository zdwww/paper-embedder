"""Slow integration tests for paper_embedder.markdown.

Exercise real pypdfium2 pre-scan, real ghostscript compression, and real
Marker model load. Opt in by:

    MARKER_INTEGRATION=1 pytest tests/test_markdown_integration.py -v -m slow

The benchmark PDFs from Plan D live at $MARKER_CORPUS_DIR (defaults to
/Users/dwww/Claude-Workspace/marker-benchmark/pdfs). Tests skip if the
corpus is not reachable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CORPUS = Path(os.environ.get(
    "MARKER_CORPUS_DIR",
    "/Users/dwww/Claude-Workspace/marker-benchmark/pdfs",
))
_SKIP_REASON = f"corpus dir not found: {CORPUS}; set MARKER_CORPUS_DIR or run Plan D corpus seed"


requires_integration = pytest.mark.skipif(
    os.environ.get("MARKER_INTEGRATION") != "1",
    reason="set MARKER_INTEGRATION=1 to run slow integration tests",
)
requires_corpus = pytest.mark.skipif(not CORPUS.exists(), reason=_SKIP_REASON)


@pytest.mark.slow
@requires_integration
@requires_corpus
def test_find_section_pages_finds_stop_keyword_on_clean_paper():
    from paper_embedder.markdown import _find_section_pages

    pdf = CORPUS / "clean_04_neuralangelo.pdf"
    if not pdf.exists():
        pytest.skip(f"corpus PDF missing: {pdf}")

    page_range, meta = _find_section_pages(pdf)

    assert len(page_range) >= 2
    assert page_range[0] == 0
    assert meta["total_pages"] > 0
    assert meta["method"] in ("stop_keyword", "fallback")


@pytest.mark.slow
@requires_integration
@requires_corpus
def test_find_section_pages_fallback_on_pathological():
    from paper_embedder.markdown import _find_section_pages

    pdf = CORPUS / "pathological_3dgs.pdf"
    if not pdf.exists():
        pytest.skip(f"corpus PDF missing: {pdf}")

    page_range, meta = _find_section_pages(pdf)

    assert len(page_range) >= 2
    assert page_range[0] == 0
    assert meta["total_pages"] > 0


@pytest.mark.slow
@requires_integration
@requires_corpus
def test_compress_pdf_shrinks_large_pdf(tmp_path):
    from paper_embedder.markdown import _compress_pdf

    src = CORPUS / "pathological_3dgs.pdf"
    if not src.exists():
        pytest.skip(f"corpus PDF missing: {src}")

    dst = tmp_path / "compressed.pdf"
    ok = _compress_pdf(src, dst)

    assert ok is True
    assert dst.exists()
    assert dst.stat().st_size > 0
    assert dst.stat().st_size < src.stat().st_size / 3


@pytest.mark.slow
@requires_integration
@requires_corpus
def test_run_marker_on_short_clean_paper():
    from paper_embedder.markdown import _find_section_pages, _run_marker

    pdf = CORPUS / "clean_03_depthreg_gs.pdf"
    if not pdf.exists():
        pytest.skip(f"corpus PDF missing: {pdf}")

    page_range, _ = _find_section_pages(pdf)
    markdown = _run_marker(pdf, page_range)

    assert isinstance(markdown, str)
    assert len(markdown) > 500
    assert "(cid:" not in markdown


@pytest.mark.slow
@requires_integration
@requires_corpus
def test_prepare_fulltext_end_to_end(tmp_path):
    """Full pipeline: pre-scan → Marker → post-trim → cache write."""
    from paper_embedder.markdown import prepare_fulltext

    pdf = CORPUS / "clean_03_depthreg_gs.pdf"
    if not pdf.exists():
        pytest.skip(f"corpus PDF missing: {pdf}")

    cache = tmp_path / "paper.md"
    out = prepare_fulltext(pdf, cache_path=cache)

    assert out is not None
    assert len(out) > 500
    assert "(cid:" not in out
    assert cache.exists()
    assert cache.read_text(encoding="utf-8") == out
