"""Text composition rules for abstract and section-fulltext vectors."""

from __future__ import annotations

from paper_embedder.types import PaperDescriptor


def compose_abstract_text(desc: PaperDescriptor) -> str:
    """Build the text used for the abstract vector.

    Papers: '{title}. {abstract}' (MRA-style). Abstract may be absent.
    Non-papers: title + creators + notes + tags joined by newlines, skipping
    empty fields (zotero-mcp-style fallback for books/webpages/notes).
    """
    if desc.item_type == "paper":
        title = desc.title.strip()
        if desc.abstract and desc.abstract.strip():
            return f"{title}. {desc.abstract.strip()}"
        return title

    # non_paper fallback
    parts = [desc.title, desc.creators, desc.notes, desc.tags]
    return "\n".join(p.strip() for p in parts if p and p.strip())


def compose_section_fulltext(desc: PaperDescriptor, section_text: str) -> str:
    """Build the text used for the section-fulltext vector.

    Prepends the title for retrieval-signal reasons (title words often complement
    methods-section language).
    """
    return f"{desc.title.strip()}\n\n{section_text.strip()}"
