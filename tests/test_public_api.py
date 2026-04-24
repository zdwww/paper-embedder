"""Verify the public API: everything users need is importable directly from paper_embedder."""


def test_public_imports():
    from paper_embedder import (
        ConfigError,
        EmbeddingMetadata,
        PaperDescriptor,
        PaperEmbedderError,
        PaperEmbeddingResult,
        ProviderConfig,
        ProviderError,
        embed_paper,
        embed_query,
        get_provider,
    )

    assert PaperDescriptor is not None
    assert ProviderConfig is not None
    assert PaperEmbeddingResult is not None
    assert EmbeddingMetadata is not None
    assert callable(get_provider)
    assert callable(embed_paper)
    assert callable(embed_query)
    assert issubclass(ConfigError, PaperEmbedderError)
    assert issubclass(ProviderError, PaperEmbedderError)


def test_markdown_submodule_is_not_exported_at_top_level():
    """paper_embedder.markdown is a submodule — callers must import it explicitly.
    It must NOT be re-exported at the top-level paper_embedder namespace."""
    import paper_embedder

    assert not hasattr(paper_embedder, "prepare_fulltext")
    assert "prepare_fulltext" not in getattr(paper_embedder, "__all__", [])


def test_end_to_end_with_stubbed_provider(tmp_path):
    """End-to-end: ProviderConfig → get_provider → embed_paper. Network is
    mocked. The caller has already produced fulltext_text (in real life this
    would come from paper_embedder.markdown.prepare_fulltext)."""
    from unittest.mock import patch

    import numpy as np

    from paper_embedder import (
        PaperDescriptor,
        ProviderConfig,
        embed_paper,
        get_provider,
    )

    class _FakeResult:
        def __init__(self, vs):
            self.embeddings = [_V(v) for v in vs]

    class _V:
        def __init__(self, vals):
            self.values = vals

    fake_client = type("FakeClient", (), {})()
    fake_client.models = type("M", (), {})()
    fake_client.models.embed_content = lambda model, contents, config: _FakeResult(
        [[0.1] * 1536 for _ in contents]
    )

    with patch("paper_embedder.providers.gemini_v2.genai.Client", return_value=fake_client):
        provider = get_provider(
            ProviderConfig(
                provider="gemini_v2",
                model_name="gemini-embedding-2-preview",
                api_key="k",
                dimension=1536,
            )
        )
        result = embed_paper(
            PaperDescriptor(
                paper_id="p1",
                title="Title",
                abstract="Abstract text.",
                fulltext_text="1. Introduction\nBody.\n2. Methods\nMore.",
                item_type="paper",
            ),
            provider,
        )

    assert isinstance(result.abstract_vec, np.ndarray)
    assert result.abstract_vec.shape == (1536,)
    assert result.fulltext_vec is not None
    assert result.metadata["fingerprint"] == provider.fingerprint()
