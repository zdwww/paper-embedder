"""paper-embedder — shared paper-embedding extraction library.

Public API:
    from paper_embedder import (
        PaperDescriptor, ProviderConfig, PaperEmbeddingResult,
        get_provider, embed_paper, embed_query,
        PaperEmbedderError, ConfigError, ProviderError, ExtractionError,
    )
"""

from paper_embedder.embedder import embed_paper, embed_query
from paper_embedder.errors import (
    ConfigError,
    ExtractionError,
    PaperEmbedderError,
    ProviderError,
)
from paper_embedder.providers import get_provider
from paper_embedder.types import (
    PaperDescriptor,
    PaperEmbeddingResult,
    ProviderConfig,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "PaperDescriptor",
    "ProviderConfig",
    "PaperEmbeddingResult",
    "get_provider",
    "embed_paper",
    "embed_query",
    "PaperEmbedderError",
    "ConfigError",
    "ProviderError",
    "ExtractionError",
]
