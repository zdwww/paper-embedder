"""embed_paper and embed_query — the public orchestration facade."""

from __future__ import annotations

import numpy as np

from paper_embedder.providers.base import Provider


def embed_query(text: str, provider: Provider) -> np.ndarray:
    """Embed a search query. Uses mode='query' so provider can apply the right
    prefix/task-type for asymmetric query/doc retrieval."""
    cleaned = text.strip()
    [vec] = provider.embed([cleaned], mode="query")
    return vec
