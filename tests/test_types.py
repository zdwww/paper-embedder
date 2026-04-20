"""Dataclass types: PaperDescriptor."""

from pathlib import Path

import pytest


def test_paper_descriptor_minimal():
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(paper_id="p1", title="Title")
    assert desc.paper_id == "p1"
    assert desc.title == "Title"
    assert desc.abstract is None
    assert desc.pdf_path is None
    assert desc.item_type == "paper"
    assert desc.creators is None
    assert desc.notes is None
    assert desc.tags is None


def test_paper_descriptor_full():
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(
        paper_id="p1",
        title="Title",
        abstract="Abstract.",
        pdf_path=Path("/tmp/p1.pdf"),
        item_type="paper",
    )
    assert desc.abstract == "Abstract."
    assert desc.pdf_path == Path("/tmp/p1.pdf")


def test_paper_descriptor_non_paper():
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(
        paper_id="note1",
        title="My Note",
        item_type="non_paper",
        creators="Alice",
        notes="Freehand text.",
        tags="tag1, tag2",
    )
    assert desc.item_type == "non_paper"
    assert desc.creators == "Alice"


def test_paper_descriptor_rejects_bad_item_type():
    from paper_embedder.types import PaperDescriptor

    with pytest.raises((ValueError, TypeError)):
        PaperDescriptor(paper_id="p1", title="T", item_type="bogus")  # type: ignore[arg-type]
