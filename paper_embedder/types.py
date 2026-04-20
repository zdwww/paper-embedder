"""Public dataclass types: PaperDescriptor, ProviderConfig, PaperEmbeddingResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from paper_embedder.errors import ConfigError

ItemType = Literal["paper", "non_paper"]
_VALID_ITEM_TYPES: tuple[str, ...] = ("paper", "non_paper")


@dataclass(frozen=True)
class PaperDescriptor:
    """Describes a paper (or non-paper Zotero item) for embedding.

    For item_type='paper': title + abstract are required for the abstract vector;
    pdf_path enables the section_fulltext vector.

    For item_type='non_paper': falls back to title+creators+notes+tags composition.
    """

    paper_id: str
    title: str
    abstract: str | None = None
    pdf_path: Path | None = None
    item_type: ItemType = "paper"
    creators: str | None = None
    notes: str | None = None
    tags: str | None = None

    def __post_init__(self) -> None:
        if self.item_type not in _VALID_ITEM_TYPES:
            raise ValueError(
                f"item_type must be one of {_VALID_ITEM_TYPES}, got {self.item_type!r}"
            )


ProviderName = Literal["gemini_v2", "openai", "huggingface"]
_VALID_PROVIDERS: tuple[str, ...] = ("gemini_v2", "openai", "huggingface")


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration used by get_provider() to construct a Provider instance."""

    provider: ProviderName
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    dimension: int | None = None

    def __post_init__(self) -> None:
        if self.provider not in _VALID_PROVIDERS:
            raise ConfigError(
                f"provider must be one of {_VALID_PROVIDERS}, got {self.provider!r}"
            )


@dataclass(frozen=True)
class PaperEmbeddingResult:
    """Returned by embed_paper(). fulltext_vec is None if no PDF or extraction failed."""

    abstract_vec: npt.NDArray[np.float32]
    fulltext_vec: npt.NDArray[np.float32] | None
    metadata: dict[str, object] = field(default_factory=dict)
