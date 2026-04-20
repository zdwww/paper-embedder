"""SectionExtractor — detect numbered section headings and stop-keywords."""


def test_find_numbered_heading_on_line():
    from paper_embedder.section_extractor import _find_heading_on_line

    assert _find_heading_on_line("1. Introduction") == (1, "Introduction")
    assert _find_heading_on_line("  2 Method  ") == (2, "Method")
    assert _find_heading_on_line("3. Related Work") == (3, "Related Work")


def test_find_heading_rejects_non_headings():
    from paper_embedder.section_extractor import _find_heading_on_line

    assert _find_heading_on_line("We present 1. a method that") is None
    assert _find_heading_on_line("   ") is None
    assert _find_heading_on_line("Introduction") is None  # no leading number
    assert _find_heading_on_line("1. a") is None  # too short


def test_is_stop_keyword_matches_known_terminators():
    from paper_embedder.section_extractor import _is_stop_keyword

    for kw in ["Results", "Experiments", "Conclusion", "Discussion",
               "References", "Acknowledgments", "Bibliography", "Appendix"]:
        assert _is_stop_keyword(kw)


def test_is_stop_keyword_case_insensitive():
    from paper_embedder.section_extractor import _is_stop_keyword

    assert _is_stop_keyword("RESULTS")
    assert _is_stop_keyword("references")


def test_is_stop_keyword_does_not_match_intro_or_methods():
    from paper_embedder.section_extractor import _is_stop_keyword

    assert not _is_stop_keyword("Introduction")
    assert not _is_stop_keyword("Method")
    assert not _is_stop_keyword("Related Work")


def test_extract_intro_and_methods_happy_path(monkeypatch, tmp_path):
    """2 pages. Page 1: intro. Page 2: methods + Results heading mid-page.
    Expected output: page1 text + page2 text UP TO 'Results' line (exclusive)."""
    from paper_embedder import section_extractor

    pages = [
        "1. Introduction\nWe introduce ideas.\n",
        "2. Methods\nWe use transformers.\n3. Results\nWe got numbers.\n",
    ]

    def fake_extract_text(path, page_numbers=None):  # noqa: ARG001
        return pages[page_numbers[0]]

    def fake_extract_pages(path):  # noqa: ARG001
        for _ in pages:
            yield object()  # page object is unused

    monkeypatch.setattr(section_extractor, "_pdfminer_extract_text", fake_extract_text)
    monkeypatch.setattr(section_extractor, "_pdfminer_extract_pages", fake_extract_pages)

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-fake")

    out = section_extractor.extract_intro_and_methods(pdf)

    assert out is not None
    assert "Introduction" in out
    assert "We introduce ideas." in out
    assert "Methods" in out
    assert "We use transformers." in out
    assert "We got numbers." not in out
    assert "3. Results" not in out


def test_extract_returns_none_when_no_headings_found(monkeypatch, tmp_path):
    from paper_embedder import section_extractor

    def fake_extract_text(path, page_numbers=None):  # noqa: ARG001
        return "Just a paragraph of flowing text with no section structure."

    def fake_extract_pages(path):  # noqa: ARG001
        yield object()

    monkeypatch.setattr(section_extractor, "_pdfminer_extract_text", fake_extract_text)
    monkeypatch.setattr(section_extractor, "_pdfminer_extract_pages", fake_extract_pages)

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-fake")

    assert section_extractor.extract_intro_and_methods(pdf) is None


def test_extract_returns_none_when_pdfminer_raises(monkeypatch, tmp_path):
    from paper_embedder import section_extractor

    def fake_extract_pages(path):  # noqa: ARG001
        raise RuntimeError("corrupted PDF")

    monkeypatch.setattr(section_extractor, "_pdfminer_extract_pages", fake_extract_pages)

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-fake")

    assert section_extractor.extract_intro_and_methods(pdf) is None


def test_extract_returns_none_when_file_missing(tmp_path):
    from paper_embedder.section_extractor import extract_intro_and_methods

    # file does not exist
    assert extract_intro_and_methods(tmp_path / "nope.pdf") is None


def test_extract_stops_scanning_after_max_pages(monkeypatch, tmp_path):
    """If paper has 20 pages but max_scan_pages=10, we only scan 10."""
    from paper_embedder import section_extractor

    calls: list[int] = []

    def fake_extract_text(path, page_numbers=None):  # noqa: ARG001
        calls.append(page_numbers[0])
        return "1. Introduction\nBody.\n"  # same content every page, no stop keyword

    def fake_extract_pages(path):  # noqa: ARG001
        for _ in range(20):
            yield object()

    monkeypatch.setattr(section_extractor, "_pdfminer_extract_text", fake_extract_text)
    monkeypatch.setattr(section_extractor, "_pdfminer_extract_pages", fake_extract_pages)

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-fake")

    section_extractor.extract_intro_and_methods(pdf, max_scan_pages=10)

    assert len(calls) <= 10
