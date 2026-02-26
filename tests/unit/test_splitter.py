"""Unit tests for MarkdownSplitter."""

from raag.splitters.markdown import MarkdownSplitter
from raag.splitters.base import SplitSection


def make_splitter():
    return MarkdownSplitter()


class TestHeadingDetection:
    """Test heading parsing at various levels."""

    def test_single_h1(self):
        text = "# Introduction\n\nSome body text here."
        sections = make_splitter().split(text)
        assert len(sections) == 1
        assert sections[0].heading_level == 1
        assert sections[0].section_title == "Introduction"
        assert sections[0].section_number is None
        assert "Some body text here." in sections[0].text

    def test_h2_with_number(self):
        text = "## 7.1 Leave Types\n\nAnnual, sick, and casual leave."
        sections = make_splitter().split(text)
        assert len(sections) == 1
        assert sections[0].heading_level == 2
        assert sections[0].section_number == "7.1"
        assert sections[0].section_title == "Leave Types"

    def test_h3_with_dotted_number(self):
        text = "### 7.9.2 Paternity Leave\n\nFathers get 5 days."
        sections = make_splitter().split(text)
        assert len(sections) == 1
        assert sections[0].heading_level == 3
        assert sections[0].section_number == "7.9.2"
        assert sections[0].section_title == "Paternity Leave"

    def test_multiple_heading_levels(self):
        text = (
            "# Chapter 1\n\nIntro.\n\n"
            "## 1.1 Section A\n\nContent A.\n\n"
            "### 1.1.1 Subsection\n\nDeep content."
        )
        sections = make_splitter().split(text)
        assert len(sections) == 3
        assert sections[0].heading_level == 1
        assert sections[1].heading_level == 2
        assert sections[2].heading_level == 3

    def test_number_with_trailing_dot(self):
        text = "## 7. Leave Policy\n\nOverview of leave."
        sections = make_splitter().split(text)
        assert sections[0].section_number == "7."
        assert sections[0].section_title == "Leave Policy"


class TestParentPath:
    """Test hierarchy breadcrumb tracking."""

    def test_h2_under_h1(self):
        text = (
            "# Parent Heading\n\nParent body.\n\n"
            "## Child Heading\n\nChild body."
        )
        sections = make_splitter().split(text)
        assert sections[0].parent_path == []
        assert sections[1].parent_path == ["Parent Heading"]

    def test_h3_under_h2_under_h1(self):
        text = (
            "# Top\n\nTop body.\n\n"
            "## Middle\n\nMiddle body.\n\n"
            "### Bottom\n\nBottom body."
        )
        sections = make_splitter().split(text)
        assert sections[0].parent_path == []
        assert sections[1].parent_path == ["Top"]
        assert sections[2].parent_path == ["Top", "Middle"]

    def test_sibling_headings_share_parent(self):
        text = (
            "# Parent\n\nParent body.\n\n"
            "## Sibling 1\n\nBody 1.\n\n"
            "## Sibling 2\n\nBody 2."
        )
        sections = make_splitter().split(text)
        assert sections[1].parent_path == ["Parent"]
        assert sections[2].parent_path == ["Parent"]

    def test_numbered_heading_in_parent_path(self):
        text = (
            "# 7. Leave Policy\n\nOverview.\n\n"
            "## 7.1 Annual Leave\n\nDetails."
        )
        sections = make_splitter().split(text)
        assert sections[1].parent_path == ["7. Leave Policy"]


class TestPreamble:
    """Test content before any heading."""

    def test_preamble_only(self):
        text = "This is plain text with no headings.\n\nAnother paragraph."
        sections = make_splitter().split(text)
        assert len(sections) == 1
        assert sections[0].heading_level == 0
        assert sections[0].section_title is None
        assert sections[0].section_number is None

    def test_preamble_then_heading(self):
        text = "Preamble text here.\n\n# The Heading\n\nBody after heading."
        sections = make_splitter().split(text)
        assert len(sections) == 2
        assert sections[0].heading_level == 0
        assert sections[0].text == "Preamble text here."
        assert sections[1].heading_level == 1

    def test_empty_preamble_skipped(self):
        text = "\n\n# Heading\n\nBody."
        sections = make_splitter().split(text)
        assert len(sections) == 1
        assert sections[0].heading_level == 1


class TestAdmonitions:
    """Test that :::note and :::warning blocks stay as body text."""

    def test_note_block_not_a_boundary(self):
        text = (
            "# Policy\n\n"
            "Some text.\n\n"
            ":::note\n"
            "Important note here.\n"
            ":::\n\n"
            "More text after note."
        )
        sections = make_splitter().split(text)
        assert len(sections) == 1
        assert "Important note here." in sections[0].text
        assert "More text after note." in sections[0].text

    def test_warning_block_not_a_boundary(self):
        text = (
            "# Rules\n\n"
            ":::warning\n"
            "Do not do this.\n"
            ":::\n\n"
            "Continued explanation."
        )
        sections = make_splitter().split(text)
        assert len(sections) == 1
        assert "Do not do this." in sections[0].text


class TestLongSectionChunking:
    """Test sub-chunking of sections exceeding max_chars."""

    def test_short_section_stays_whole(self):
        text = "# Title\n\nShort body."
        sections = make_splitter().split(text, max_chars=1000)
        assert len(sections) == 1
        assert sections[0].chunk_index == 0
        assert sections[0].total_chunks == 1

    def test_long_section_splits_at_paragraphs(self):
        para = "A" * 400
        text = f"# Title\n\n{para}\n\n{para}\n\n{para}"
        sections = make_splitter().split(text, max_chars=500)
        assert len(sections) > 1
        for s in sections:
            assert s.section_title == "Title"
            assert s.heading_level == 1

    def test_chunk_indices_sequential(self):
        para = "B" * 300
        text = f"# H\n\n{para}\n\n{para}\n\n{para}"
        sections = make_splitter().split(text, max_chars=400)
        total = sections[0].total_chunks
        for i, s in enumerate(sections):
            assert s.chunk_index == i
            assert s.total_chunks == total

    def test_continued_suffix_on_subsequent_chunks(self):
        para = "C" * 300
        text = f"# My Heading\n\n{para}\n\n{para}\n\n{para}"
        sections = make_splitter().split(text, max_chars=400)
        if len(sections) > 1:
            assert "# My Heading" in sections[0].text
            assert "(continued)" in sections[1].text


class TestEdgeCases:
    """Edge cases and empty input handling."""

    def test_empty_string(self):
        sections = make_splitter().split("")
        assert sections == []

    def test_whitespace_only(self):
        sections = make_splitter().split("   \n\n   ")
        assert sections == []

    def test_heading_without_body_skipped(self):
        text = "# Just a Heading"
        sections = make_splitter().split(text)
        assert len(sections) == 0

    def test_consecutive_headings_no_body_skipped(self):
        text = "# First\n# Second\n# Third"
        sections = make_splitter().split(text)
        assert len(sections) == 0

    def test_heading_with_only_whitespace_body_skipped(self):
        text = "# Title\n\n   \n\n## Next\n\nBody."
        sections = make_splitter().split(text)
        # Title heading has no real body, gets skipped
        assert len(sections) == 1
        assert sections[0].heading_level == 2
        assert sections[0].section_title == "Next"
