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
