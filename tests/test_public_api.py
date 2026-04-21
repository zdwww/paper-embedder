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


def test_end_to_end_with_stubbed_provider_and_extractor(monkeypatch, tmp_path):
    """End-to-end: ProviderConfig → get_provider → embed_paper. Network + PDF
    extraction both mocked so this test runs without external deps."""
    from unittest.mock import patch

    import numpy as np

    from paper_embedder import (
        PaperDescriptor,
        ProviderConfig,
        embed_paper,
        get_provider,
        section_extractor,
    )
    from paper_embedder import embedder as embedder_mod

    monkeypatch.setattr(
        section_extractor,
        "extract_intro_and_methods",
        lambda p, max_scan_pages=10: "1. Introduction\nBody.\n2. Methods\nMore.",
    )
    monkeypatch.setattr(
        embedder_mod,
        "extract_intro_and_methods",
        section_extractor.extract_intro_and_methods,
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

    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-fake")

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
                pdf_path=pdf,
                item_type="paper",
            ),
            provider,
        )

    assert isinstance(result.abstract_vec, np.ndarray)
    assert result.abstract_vec.shape == (1536,)
    assert result.fulltext_vec is not None
    assert result.metadata["fingerprint"] == provider.fingerprint()
