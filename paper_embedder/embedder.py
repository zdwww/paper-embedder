"""embed_paper and embed_query — the public orchestration facade."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paper_embedder.composers import compose_abstract_text, compose_section_fulltext
from paper_embedder.providers.base import Provider
from paper_embedder.section_extractor import extract_intro_and_methods
from paper_embedder.types import PaperDescriptor, PaperEmbeddingResult


def embed_query(text: str, provider: Provider) -> np.ndarray:
    """Embed a search query. Uses mode='query' so provider can apply the right
    prefix/task-type for asymmetric query/doc retrieval."""
    cleaned = text.strip()
    [vec] = provider.embed([cleaned], mode="query")
    return vec


def embed_paper(paper: PaperDescriptor, provider: Provider) -> PaperEmbeddingResult:
    """Build both vectors for a paper (or single vector for non_paper items).

    Paper items with a pdf_path:
      - abstract_vec from 'title. abstract' text
      - fulltext_vec from 'title + intro+methods' text (None if extraction fails)

    Paper items without pdf_path:
      - abstract_vec only; fulltext_vec = None

    Non-paper items:
      - abstract_vec from concat(title, creators, notes, tags); fulltext_vec = None

    Never raises for extraction failures (returns None fulltext_vec). Provider
    failures propagate as ProviderError.
    """
    abstract_text = compose_abstract_text(paper)
    [abstract_vec] = provider.embed([abstract_text], mode="document")

    fulltext_vec: np.ndarray | None = None
    if paper.item_type == "paper" and paper.pdf_path is not None:
        section_text = extract_intro_and_methods(paper.pdf_path)
        if section_text is not None:
            fulltext_text = compose_section_fulltext(paper, section_text)
            [fulltext_vec] = provider.embed([fulltext_text], mode="document")

    return PaperEmbeddingResult(
        abstract_vec=abstract_vec,
        fulltext_vec=fulltext_vec,
        metadata={
            "model": provider.name,
            "dim": provider.dim,
            "fingerprint": provider.fingerprint(),
        },
    )
