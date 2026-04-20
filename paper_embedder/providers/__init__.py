"""Provider factory — maps ProviderConfig to a concrete Provider instance."""

from paper_embedder.errors import ConfigError
from paper_embedder.providers.base import EmbedMode, Provider
from paper_embedder.providers.gemini_v2 import GeminiV2Provider
from paper_embedder.types import ProviderConfig

_GEMINI_V2_DEFAULT_DIM = 1536


def get_provider(config: ProviderConfig) -> Provider:
    if config.provider == "gemini_v2":
        if not config.api_key:
            raise ConfigError("Gemini v2 provider requires api_key")
        dim = config.dimension if config.dimension is not None else _GEMINI_V2_DEFAULT_DIM
        return GeminiV2Provider(
            api_key=config.api_key,
            model_name=config.model_name,
            dim=dim,
            base_url=config.base_url,
        )
    if config.provider in ("openai", "huggingface"):
        raise ConfigError(
            f"provider {config.provider!r} is not yet implemented (Plan D, paper-embedder v0.3)"
        )
    raise ConfigError(f"unknown provider: {config.provider!r}")


__all__ = ["Provider", "EmbedMode", "get_provider"]
