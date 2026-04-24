"""Smoke test — verifies the package imports cleanly."""


def test_package_imports():
    import paper_embedder

    assert paper_embedder.__version__ == "0.3.0"
