"""Tests for the post-LLM citation validator (TDD).

The validator runs between Claude's response and the UI render. It must:
- preserve valid citations untouched
- normalize comma-grouped citations like "[1, 2]" into "[1][2]" so the strict
  linkifier regex in main.py can render them as anchor chips
- strip citations referring to source numbers that don't exist, and report a
  warning for each unique offender
- strip empty/whitespace-only brackets
- warn when the answer contains any citation but no sources were retrieved
"""

from src.rag import Source, validate_citations


def _src(n: int) -> Source:
    return Source(
        number=n,
        file_id=f"f{n}",
        file_name=f"name{n}",
        mime_type="text/plain",
        web_view_url=f"https://docs.google.com/document/d/{n}",
    )


class TestValidateCitations:
    def test_valid_single_citations_pass_through(self):
        sources = [_src(1), _src(2)]
        text, warns = validate_citations("Slack is approved [1]. Zoom too [2].", sources)
        assert text == "Slack is approved [1]. Zoom too [2]."
        assert warns == []

    def test_normalizes_comma_grouped_to_bracket_chain(self):
        """The render regex in main.py matches '[N]' but not '[N, M]' — the
        validator must rewrite the latter so the linkifier sees individual brackets."""
        sources = [_src(1), _src(2)]
        text, warns = validate_citations("Both apps [1, 2].", sources)
        assert text == "Both apps [1][2]."
        assert warns == []

    def test_normalizes_comma_no_space(self):
        sources = [_src(1), _src(2), _src(3)]
        text, _ = validate_citations("All three [1,2,3] cited.", sources)
        assert text == "All three [1][2][3] cited."

    def test_keeps_already_chained_brackets(self):
        sources = [_src(1), _src(2)]
        text, warns = validate_citations("Chained [1][2] form.", sources)
        assert text == "Chained [1][2] form."
        assert warns == []

    def test_out_of_range_citation_stripped_with_warning(self):
        sources = [_src(1), _src(2)]
        text, warns = validate_citations("Approved [5].", sources)
        assert "[5]" not in text
        assert text == "Approved ."
        assert len(warns) == 1
        assert "[5]" in warns[0]

    def test_zero_citation_stripped(self):
        sources = [_src(1)]
        text, warns = validate_citations("Bad [0] cite.", sources)
        assert "[0]" not in text
        assert any("[0]" in w for w in warns)

    def test_duplicate_invalid_warnings_deduped(self):
        sources = [_src(1)]
        _, warns = validate_citations("[7] and [7] and [7] again.", sources)
        # one unique offender → one warning
        assert len(warns) == 1
        assert "[7]" in warns[0]

    def test_distinct_invalid_get_distinct_warnings(self):
        sources = [_src(1)]
        _, warns = validate_citations("Bad [4] and [9].", sources)
        assert len(warns) == 2

    def test_mixed_valid_and_invalid_in_group(self):
        sources = [_src(1), _src(2)]
        text, warns = validate_citations("Mixed [1, 7].", sources)
        # [1] survives, [7] stripped
        assert "[1]" in text
        assert "[7]" not in text
        assert "[1, 7]" not in text  # original grouped form is gone
        assert len(warns) == 1

    def test_no_citations_passes_through_unchanged(self):
        sources = [_src(1)]
        text, warns = validate_citations("No citations here at all.", sources)
        assert text == "No citations here at all."
        assert warns == []

    def test_whitespace_only_brackets_stripped(self):
        sources = [_src(1)]
        # "[ ]" has only whitespace — no digits, so it's a malformed citation. Strip.
        text, _ = validate_citations("Weird [ ] case.", sources)
        assert "[ ]" not in text

    def test_citation_when_no_sources_warns(self):
        text, warns = validate_citations("Has [1] but no sources retrieved.", sources=[])
        assert "[1]" not in text
        assert any("no sources" in w.lower() for w in warns)

    def test_empty_input_returns_empty(self):
        text, warns = validate_citations("", sources=[_src(1)])
        assert text == ""
        assert warns == []

    def test_grouped_with_invalid_stripped_completely(self):
        """A group where every entry is invalid gets stripped entirely (no empty brackets left)."""
        sources = [_src(1)]
        text, _ = validate_citations("Bogus [5, 6, 7].", sources)
        assert "[5" not in text
        assert "[6" not in text
        assert "[7" not in text
        # The whole bracket is gone (replaced by empty), leaving "Bogus ."
        assert text == "Bogus ."
