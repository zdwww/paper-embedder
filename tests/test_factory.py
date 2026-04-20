"""get_provider(): factory that turns ProviderConfig into a concrete Provider."""

import pytest


def test_get_provider_gemini_v2_returns_gemini_instance():
    from paper_embedder.providers import get_provider
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider
    from paper_embedder.types import ProviderConfig

    cfg = ProviderConfig(
        provider="gemini_v2",
        model_name="gemini-embedding-2-preview",
        api_key="k",
        dimension=1536,
    )
    p = get_provider(cfg)
    assert isinstance(p, GeminiV2Provider)
    assert p.dim == 1536


def test_get_provider_gemini_v2_defaults_dimension_to_1536():
    from paper_embedder.providers import get_provider
    from paper_embedder.types import ProviderConfig

    cfg = ProviderConfig(
        provider="gemini_v2",
        model_name="gemini-embedding-2-preview",
        api_key="k",
    )
    p = get_provider(cfg)
    assert p.dim == 1536


def test_get_provider_gemini_v2_missing_api_key_raises():
    from paper_embedder.errors import ConfigError
    from paper_embedder.providers import get_provider
    from paper_embedder.types import ProviderConfig

    cfg = ProviderConfig(
        provider="gemini_v2",
        model_name="gemini-embedding-2-preview",
        api_key=None,
    )
    with pytest.raises(ConfigError, match="api_key"):
        get_provider(cfg)


def test_get_provider_openai_not_implemented_yet():
    from paper_embedder.errors import ConfigError
    from paper_embedder.providers import get_provider
    from paper_embedder.types import ProviderConfig

    cfg = ProviderConfig(provider="openai", model_name="text-embedding-3-small", api_key="k")
    with pytest.raises(ConfigError, match="not yet implemented"):
        get_provider(cfg)


def test_get_provider_huggingface_not_implemented_yet():
    from paper_embedder.errors import ConfigError
    from paper_embedder.providers import get_provider
    from paper_embedder.types import ProviderConfig

    cfg = ProviderConfig(provider="huggingface", model_name="Qwen/Qwen3-Embedding-0.6B")
    with pytest.raises(ConfigError, match="not yet implemented"):
        get_provider(cfg)
