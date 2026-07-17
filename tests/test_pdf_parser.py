"""
Tests for rct_reviewer.core.pdf_parser.PDFParser
"""

import pytest
from rct_reviewer.core.pdf_parser import PDFParser


@pytest.fixture(scope="module")
def parser():
    """PDFParser is expensive to create (loads spacy model).
    Share one instance across all tests in this module."""
    return PDFParser()


class TestPDFParser:
    def test_parse_returns_expected_keys(self, parser, dummy_pdf_bytes):
        result = parser.parse(dummy_pdf_bytes)
        assert "text" in result
        assert "sentences" in result
        assert "title" in result
        assert "abstract" in result

    def test_text_is_nonempty(self, parser, dummy_pdf_bytes):
        result = parser.parse(dummy_pdf_bytes)
        assert len(result["text"]) > 0

    def test_sentences_extracted(self, parser, dummy_pdf_bytes):
        result = parser.parse(dummy_pdf_bytes)
        assert len(result["sentences"]) > 0

    def test_sentence_structure(self, parser, dummy_pdf_bytes):
        result = parser.parse(dummy_pdf_bytes)
        for sent in result["sentences"]:
            assert "text" in sent
            assert "start" in sent
            assert "end" in sent
            assert isinstance(sent["text"], str)
            assert sent["end"] >= sent["start"]

    def test_title_is_first_sentence(self, parser, dummy_pdf_bytes):
        result = parser.parse(dummy_pdf_bytes)
        if result["sentences"]:
            assert result["title"] == result["sentences"][0]["text"]

    def test_abstract_is_truncated_text(self, parser, dummy_pdf_bytes):
        result = parser.parse(dummy_pdf_bytes)
        assert result["abstract"] == result["text"][:3000]

    def test_text_contains_known_content(self, parser, dummy_pdf_bytes):
        result = parser.parse(dummy_pdf_bytes)
        assert "randomized" in result["text"].lower()
        assert "drug x" in result["text"].lower()

    def test_empty_pdf(self, parser):
        """A PDF with no text should return empty strings/lists."""
        import fitz
        import io
        doc = fitz.open()
        doc.new_page(width=612, height=792)
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        buf.seek(0)
        result = parser.parse(buf.getvalue())

        assert isinstance(result["text"], str)
        assert isinstance(result["sentences"], list)