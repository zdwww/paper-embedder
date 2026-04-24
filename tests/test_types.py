"""Dataclass types: PaperDescriptor."""

import pytest


def test_paper_descriptor_minimal():
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(paper_id="p1", title="Title")
    assert desc.paper_id == "p1"
    assert desc.title == "Title"
    assert desc.abstract is None
    assert desc.fulltext_text is None
    assert desc.item_type == "paper"
    assert desc.creators is None
    assert desc.notes is None
    assert desc.tags is None


def test_paper_descriptor_full():
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(
        paper_id="p1",
        title="Title",
        abstract="Abstract.",
        fulltext_text="1. Introduction\nbody\n2. Method\nmore",
        item_type="paper",
    )
    assert desc.abstract == "Abstract."
    assert desc.fulltext_text is not None
    assert "1. Introduction" in desc.fulltext_text


def test_paper_descriptor_non_paper():
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(
        paper_id="note1",
        title="My Note",
        item_type="non_paper",
        creators="Alice",
        notes="Freehand text.",
        tags="tag1, tag2",
    )
    assert desc.item_type == "non_paper"
    assert desc.creators == "Alice"


def test_paper_descriptor_rejects_bad_item_type():
    from paper_embedder.types import PaperDescriptor

    with pytest.raises((ValueError, TypeError)):
        PaperDescriptor(paper_id="p1", title="T", item_type="bogus")  # type: ignore[arg-type]


def test_provider_config_gemini_v2():
    from paper_embedder.types import ProviderConfig

    cfg = ProviderConfig(
        provider="gemini_v2",
        model_name="gemini-embedding-2-preview",
        api_key="sk-test",
        dimension=1536,
    )
    assert cfg.provider == "gemini_v2"
    assert cfg.model_name == "gemini-embedding-2-preview"
    assert cfg.api_key == "sk-test"
    assert cfg.base_url is None
    assert cfg.dimension == 1536


def test_provider_config_defaults_dimension_to_none():
    from paper_embedder.types import ProviderConfig

    cfg = ProviderConfig(provider="gemini_v2", model_name="foo", api_key="k")
    assert cfg.dimension is None


def test_provider_config_rejects_unknown_provider():
    from paper_embedder.errors import ConfigError
    from paper_embedder.types import ProviderConfig

    with pytest.raises(ConfigError, match="provider"):
        ProviderConfig(provider="bogus", model_name="foo", api_key="k")  # type: ignore[arg-type]


def test_paper_embedding_result_with_both_vectors():
    import numpy as np

    from paper_embedder.types import EmbeddingMetadata, PaperEmbeddingResult

    meta: EmbeddingMetadata = {
        "model": "gemini-embedding-2-preview",
        "dim": 1536,
        "fingerprint": "sha256:abc",
    }
    result = PaperEmbeddingResult(
        abstract_vec=np.zeros(1536, dtype=np.float32),
        fulltext_vec=np.ones(1536, dtype=np.float32),
        metadata=meta,
    )
    assert result.abstract_vec.shape == (1536,)
    assert result.fulltext_vec is not None
    assert result.fulltext_vec.shape == (1536,)
    assert result.metadata["model"] == "gemini-embedding-2-preview"
    assert result.metadata["fingerprint"] == "sha256:abc"


def test_paper_embedding_result_fulltext_optional():
    import numpy as np

    from paper_embedder.types import EmbeddingMetadata, PaperEmbeddingResult

    meta: EmbeddingMetadata = {"model": "foo", "dim": 1536, "fingerprint": "fp"}
    result = PaperEmbeddingResult(
        abstract_vec=np.zeros(1536, dtype=np.float32),
        fulltext_vec=None,
        metadata=meta,
    )
    assert result.fulltext_vec is None
