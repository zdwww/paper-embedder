"""embed_paper and embed_query — the public orchestration facade."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from paper_embedder.composers import compose_abstract_text, compose_section_fulltext
from paper_embedder.providers.base import Provider
from paper_embedder.types import EmbeddingMetadata, PaperDescriptor, PaperEmbeddingResult


def embed_query(text: str, provider: Provider) -> npt.NDArray[np.float32]:
    """Embed a search query. Uses mode='query' so provider can apply the right
    prefix/task-type for asymmetric query/doc retrieval."""
    cleaned = text.strip()
    [vec] = provider.embed([cleaned], mode="query")
    return vec


def embed_paper(paper: PaperDescriptor, provider: Provider) -> PaperEmbeddingResult:
    """Build both vectors for a paper (or single vector for non_paper items).

    Paper items with fulltext_text:
      - abstract_vec from 'title. abstract' text
      - fulltext_vec from 'title + fulltext_text' text

    Paper items without fulltext_text (or fulltext_text=''):
      - abstract_vec only; fulltext_vec = None

    Non-paper items:
      - abstract_vec from concat(title, creators, notes, tags); fulltext_vec = None
      - fulltext_text is ignored even if present

    Pure function: no filesystem writes, no PDF parsing, no network I/O beyond
    the provider's own retries. Provider failures propagate as ProviderError.
    """
    abstract_text = compose_abstract_text(paper)
    [abstract_vec] = provider.embed([abstract_text], mode="document")

    fulltext_vec: npt.NDArray[np.float32] | None = None
    if paper.item_type == "paper" and paper.fulltext_text:
        fulltext_text = compose_section_fulltext(paper, paper.fulltext_text)
        [fulltext_vec] = provider.embed([fulltext_text], mode="document")

    metadata: EmbeddingMetadata = {
        "model": provider.name,
        "dim": provider.dim,
        "fingerprint": provider.fingerprint(),
    }
    return PaperEmbeddingResult(
        abstract_vec=abstract_vec,
        fulltext_vec=fulltext_vec,
        metadata=metadata,
    )
