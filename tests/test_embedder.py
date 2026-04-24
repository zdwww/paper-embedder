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


def test_embed_paper_with_fulltext_returns_both_vectors():
    from paper_embedder.embedder import embed_paper
    from paper_embedder.types import PaperDescriptor

    provider = _fake_provider(dim=1536)
    result = embed_paper(
        PaperDescriptor(
            paper_id="p1",
            title="T",
            abstract="A.",
            fulltext_text="1. Introduction\nBody of intro.\n2. Methods\nStuff.",
            item_type="paper",
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


def test_embed_paper_without_fulltext_returns_abstract_only():
    from paper_embedder.embedder import embed_paper
    from paper_embedder.types import PaperDescriptor

    provider = _fake_provider()
    result = embed_paper(
        PaperDescriptor(
            paper_id="p1",
            title="T",
            abstract="A.",
            fulltext_text=None,
            item_type="paper",
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
            paper_id="n1",
            title="Webpage",
            item_type="non_paper",
            creators="Alice",
            notes="notes",
            tags="t1",
        ),
        provider,
    )

    assert result.fulltext_vec is None
    passed = provider.embed.call_args.args[0][0]
    assert "Webpage" in passed
    assert "Alice" in passed
    assert "notes" in passed
    assert "t1" in passed


def test_embed_paper_non_paper_ignores_fulltext_text():
    """Even if a non_paper item is constructed with fulltext_text (e.g. caller
    mistakenly passed it), embed_paper MUST NOT build a fulltext_vec for it."""
    from paper_embedder.embedder import embed_paper
    from paper_embedder.types import PaperDescriptor

    provider = _fake_provider()
    result = embed_paper(
        PaperDescriptor(
            paper_id="n1",
            title="Webpage",
            item_type="non_paper",
            fulltext_text="this should be ignored",
        ),
        provider,
    )

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
