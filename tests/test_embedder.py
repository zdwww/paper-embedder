"""embed_query and embed_paper orchestration."""

from unittest.mock import MagicMock

import numpy as np


def _fake_provider(dim: int = 1536) -> MagicMock:
    p = MagicMock()
    p.name = "gemini_v2"
    p.dim = dim
    p.max_input_tokens = 7980
    p.truncate.side_effect = lambda t: t
    p.embed.side_effect = lambda texts, *, mode: [
        np.full(dim, 0.1 if mode == "document" else 0.2, dtype=np.float32) for _ in texts
    ]
    p.fingerprint.return_value = "deadbeef"
    return p


def test_embed_query_returns_single_vector():
    from paper_embedder.embedder import embed_query

    provider = _fake_provider(dim=1536)
    vec = embed_query("what is attention?", provider)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1536,)
    assert vec.dtype == np.float32


def test_embed_query_uses_query_mode():
    from paper_embedder.embedder import embed_query

    provider = _fake_provider(dim=1536)
    embed_query("what is attention?", provider)

    call = provider.embed.call_args
    assert call.kwargs["mode"] == "query"
    assert call.args[0] == ["what is attention?"]


def test_embed_query_strips_surrounding_whitespace():
    from paper_embedder.embedder import embed_query

    provider = _fake_provider()
    embed_query("   hello   ", provider)

    assert provider.embed.call_args.args[0] == ["hello"]


def test_embed_paper_with_pdf_returns_both_vectors(monkeypatch, tmp_path):
    from pathlib import Path

    from paper_embedder import embedder, section_extractor
    from paper_embedder.embedder import embed_paper
    from paper_embedder.types import PaperDescriptor

    # Stub section extraction so we don't need a real PDF.
    monkeypatch.setattr(
        section_extractor,
        "extract_intro_and_methods",
        lambda p, max_scan_pages=10: "1. Introduction\nBody of intro.\n2. Methods\nStuff.",
    )
    # Ensure the embedder module also calls the same stub (it imports at top-level).
    monkeypatch.setattr(
        embedder,
        "extract_intro_and_methods",
        section_extractor.extract_intro_and_methods,
    )

    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-fake")

    provider = _fake_provider(dim=1536)
    result = embed_paper(
        PaperDescriptor(
            paper_id="p1", title="T", abstract="A.", pdf_path=pdf, item_type="paper",
        ),
        provider,
    )

    assert result.abstract_vec.shape == (1536,)
    assert result.fulltext_vec is not None
    assert result.fulltext_vec.shape == (1536,)
    assert result.metadata["model"] == "gemini_v2"
    assert result.metadata["dim"] == 1536
    assert result.metadata["fingerprint"] == "deadbeef"
    # Provider called twice: once for abstract, once for fulltext
    assert provider.embed.call_count == 2


def test_embed_paper_without_pdf_returns_abstract_only():
    from paper_embedder.embedder import embed_paper
    from paper_embedder.types import PaperDescriptor

    provider = _fake_provider()
    result = embed_paper(
        PaperDescriptor(
            paper_id="p1", title="T", abstract="A.", pdf_path=None, item_type="paper",
        ),
        provider,
    )

    assert result.abstract_vec.shape == (1536,)
    assert result.fulltext_vec is None
    assert provider.embed.call_count == 1


def test_embed_paper_non_paper_returns_abstract_only():
    from paper_embedder.embedder import embed_paper
    from paper_embedder.types import PaperDescriptor

    provider = _fake_provider()
    result = embed_paper(
        PaperDescriptor(
            paper_id="n1", title="Webpage", item_type="non_paper",
            creators="Alice", notes="notes", tags="t1",
        ),
        provider,
    )

    assert result.fulltext_vec is None
    # Check the concatenated non_paper text was passed to embed
    passed = provider.embed.call_args.args[0][0]
    assert "Webpage" in passed
    assert "Alice" in passed
    assert "notes" in passed
    assert "t1" in passed


def test_embed_paper_extraction_failure_returns_abstract_only(monkeypatch, tmp_path):
    from paper_embedder import embedder, section_extractor
    from paper_embedder.embedder import embed_paper
    from paper_embedder.types import PaperDescriptor

    # Section extraction returns None — simulating corrupted PDF
    monkeypatch.setattr(section_extractor, "extract_intro_and_methods", lambda p, max_scan_pages=10: None)
    monkeypatch.setattr(embedder, "extract_intro_and_methods", section_extractor.extract_intro_and_methods)

    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"%PDF-fake")

    provider = _fake_provider()
    result = embed_paper(
        PaperDescriptor(paper_id="p1", title="T", abstract="A.", pdf_path=pdf, item_type="paper"),
        provider,
    )

    assert result.abstract_vec.shape == (1536,)
    assert result.fulltext_vec is None
    assert provider.embed.call_count == 1


def test_embed_paper_propagates_provider_error():
    import pytest

    from paper_embedder.embedder import embed_paper
    from paper_embedder.errors import ProviderError
    from paper_embedder.types import PaperDescriptor

    provider = _fake_provider()
    provider.embed.side_effect = ProviderError("upstream failure")

    with pytest.raises(ProviderError, match="upstream failure"):
        embed_paper(PaperDescriptor(paper_id="p1", title="T"), provider)


def test_embed_paper_does_not_modify_input_pdf(tmp_path):
    """Bottom-line invariant (spec §1): never modify source PDFs. Check SHA-256 +
    mtime + size before and after a full embed_paper call."""
    import hashlib
    import os

    from paper_embedder.embedder import embed_paper
    from paper_embedder.types import PaperDescriptor

    # Minimal real PDF content
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\n"
        b"xref\n0 3\n0000000000 65535 f\n0000000009 00000 n\n"
        b"0000000055 00000 n\ntrailer<</Size 3/Root 1 0 R>>\n"
        b"startxref\n103\n%%EOF\n"
    )
    before_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    before_stat = pdf.stat()

    provider = _fake_provider()
    # extract will return None on this minimal PDF — that's fine, we only care
    # that pdf contents are unchanged.
    embed_paper(
        PaperDescriptor(paper_id="p1", title="T", abstract="A.", pdf_path=pdf, item_type="paper"),
        provider,
    )

    after_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    after_stat = pdf.stat()

    assert before_sha == after_sha, "PDF contents must be unchanged"
    assert before_stat.st_size == after_stat.st_size, "PDF size must be unchanged"
    assert before_stat.st_mtime_ns == after_stat.st_mtime_ns, "PDF mtime must be unchanged"
