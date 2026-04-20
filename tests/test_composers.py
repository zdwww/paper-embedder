"""Text composition rules for abstract vector and section-fulltext vector."""


def test_compose_abstract_paper_with_abstract():
    from paper_embedder.composers import compose_abstract_text
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(
        paper_id="p1", title="Attention Is All You Need",
        abstract="We propose the Transformer.",
    )
    assert compose_abstract_text(desc) == "Attention Is All You Need. We propose the Transformer."


def test_compose_abstract_paper_without_abstract():
    from paper_embedder.composers import compose_abstract_text
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(paper_id="p1", title="Lonely Title")
    assert compose_abstract_text(desc) == "Lonely Title"


def test_compose_abstract_paper_strips_whitespace():
    from paper_embedder.composers import compose_abstract_text
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(paper_id="p1", title="  Title  ", abstract="  Body.  ")
    assert compose_abstract_text(desc) == "Title. Body."


def test_compose_abstract_non_paper_full():
    from paper_embedder.composers import compose_abstract_text
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(
        paper_id="n1", title="My Webpage",
        item_type="non_paper",
        creators="Alice, Bob",
        notes="Interesting content.",
        tags="ml, nlp",
    )
    out = compose_abstract_text(desc)
    assert out == "My Webpage\nAlice, Bob\nInteresting content.\nml, nlp"


def test_compose_abstract_non_paper_skips_missing_fields():
    from paper_embedder.composers import compose_abstract_text
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(
        paper_id="n1", title="Bare Item", item_type="non_paper", creators=None, notes=None, tags=None,
    )
    assert compose_abstract_text(desc) == "Bare Item"


def test_compose_abstract_deterministic():
    from paper_embedder.composers import compose_abstract_text
    from paper_embedder.types import PaperDescriptor

    desc = PaperDescriptor(paper_id="p1", title="T", abstract="A.")
    assert compose_abstract_text(desc) == compose_abstract_text(desc)
