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
