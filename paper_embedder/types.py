"""Public dataclass types: PaperDescriptor, ProviderConfig, PaperEmbeddingResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


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
