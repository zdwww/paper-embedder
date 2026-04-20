"""Exception hierarchy: all package-specific errors inherit from PaperEmbedderError."""

import pytest


def test_exception_hierarchy():
    from paper_embedder.errors import (
        ConfigError,
        ExtractionError,
        PaperEmbedderError,
        ProviderError,
    )

    assert issubclass(ConfigError, PaperEmbedderError)
    assert issubclass(ProviderError, PaperEmbedderError)
    assert issubclass(ExtractionError, PaperEmbedderError)
    assert issubclass(PaperEmbedderError, Exception)


def test_errors_can_carry_messages():
    from paper_embedder.errors import ConfigError

    with pytest.raises(ConfigError, match="bad config"):
        raise ConfigError("bad config")
