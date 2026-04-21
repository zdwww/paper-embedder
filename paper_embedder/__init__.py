"""paper-embedder — shared paper-embedding extraction library.

Public API:
    from paper_embedder import (
        PaperDescriptor, ProviderConfig, PaperEmbeddingResult, EmbeddingMetadata,
        get_provider, embed_paper, embed_query,
        PaperEmbedderError, ConfigError, ProviderError,
    )
"""

from paper_embedder.embedder import embed_paper, embed_query
from paper_embedder.errors import (
    ConfigError,
    PaperEmbedderError,
    ProviderError,
)
from paper_embedder.providers import get_provider
from paper_embedder.types import (
    EmbeddingMetadata,
    PaperDescriptor,
    PaperEmbeddingResult,
    ProviderConfig,
)

__version__ = "0.2.3"

__all__ = [
    "__version__",
    "PaperDescriptor",
    "ProviderConfig",
    "PaperEmbeddingResult",
    "EmbeddingMetadata",
    "get_provider",
    "embed_paper",
    "embed_query",
    "PaperEmbedderError",
    "ConfigError",
    "ProviderError",
]
