"""Provider protocol — the interface every embedding provider must satisfy."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

EmbedMode = Literal["document", "query"]


@runtime_checkable
class Provider(Protocol):
    """Any class with these attributes+methods is a valid Provider."""

    name: str
    dim: int
    max_input_tokens: int

    def embed(self, texts: list[str], *, mode: EmbedMode) -> list[np.ndarray]:
        """Embed a batch of texts. `mode` selects document- vs. query-side handling
        (e.g. Gemini v2 uses different prefixes, OpenAI ignores, etc.).
        Returns one np.ndarray of shape (dim,) per input text.
        Retries transient errors internally, up to 3 times; raises ProviderError
        on permanent failure.
        """
        ...

    def truncate(self, text: str) -> str:
        """Truncate text so the provider's embed() call stays within max_input_tokens."""
        ...

    def fingerprint(self) -> str:
        """Stable sha256 hex digest of (provider name, model, dim, task-type convention).
        Used by callers to detect config changes and invalidate stored vectors.
        """
        ...
