"""Exception hierarchy for paper-embedder."""


class PaperEmbedderError(Exception):
    """Base class for all paper-embedder errors."""


class ConfigError(PaperEmbedderError):
    """Raised when ProviderConfig is invalid (bad provider name, missing API key, etc.)."""


class ProviderError(PaperEmbedderError):
    """Raised when an embedding provider's API call fails after retries."""


class ExtractionError(PaperEmbedderError):
    """Raised internally by SectionExtractor; normally caught and converted to fulltext_vec=None."""
