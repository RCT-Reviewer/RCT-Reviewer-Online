"""
Tests for rct_reviewer.app2 — every function except the Streamlit main() UI.

Streamlit is mocked in conftest.py before this file is imported, so
module-level calls like st.set_page_config() are no-ops.
"""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pandas as pd
import pytest

# streamlit is already mocked by conftest.py
from rct_reviewer.app2 import (
    BIAS_COLORS,
    BIAS_LETTERS,
    PICO_COLORS,
    PICO_LETTERS,
    _apply_highlight_and_get_anchor,
    _clean_text_for_pdf,
    _expand_to_lines,
    _find_next_non_overlapping_x,
    _format_eta,
    _insert_evidence_textbox,
    _normalize_text,
    _wrap_text,
    create_bias_evidence_pdf,
    create_bias_highlighted_pdf,
    create_pico_evidence_pdf,
    create_pico_highlighted_pdf,
    download_models,
    export_to_csv,
    export_to_json,
    find_text_areas,
    js_escape,
    load_models_with_progress,
)


# ======================================================================
# 1. Utility / pure functions
# ======================================================================


class TestFormatEta:
    def test_negative(self):
        assert _format_eta(-1) == "calculating..."

    def test_inf(self):
        assert _format_eta(float("inf")) == "calculating..."

    def test_nan(self):
        assert _format_eta(float("nan")) == "calculating..."

    def test_less_than_one_second(self):
        assert _format_eta(0.5) == "<1s"

    def test_seconds(self):
        assert _format_eta(30) == "~30s"

    def test_minutes_and_seconds(self):
        assert _format_eta(125) == "~2m 5s"

    def test_hours_and_minutes(self):
        assert _format_eta(3725) == "~1h 2m"

    def test_exact_minute(self):
        assert _format_eta(60) == "~1m 0s"

    def test_zero(self):
        assert _format_eta(0) == "<1s"


class TestNormalizeText:
    def test_collapses_whitespace(self):
        assert _normalize_text("  hello   world  ") == "hello world"

    def test_ligatures(self):
        assert _normalize_text("\ufb01nancial \ufb02uid") == "financial fluid"

    def test_ff_ligature(self):
        assert _normalize_text("\ufb00ect") == "ffect"

    def test_ffi_ligature(self):
        assert _normalize_text("\ufb03rm") == "ffirm"

    def test_ffl_ligature(self):
        assert _normalize_text("\ufb04ected") == "fflected"

    def test_dashes(self):
        for ch in "\u2010\u2011\u2012\u2013\u2014":
            assert "-" in _normalize_text(f"word{ch}part")

    def test_right_quote(self):
        assert _normalize_text("\u2019s") == "'s"

    def test_prime(self):
        assert _normalize_text("\u2032") == "'"

    def test_double_prime(self):
        assert _normalize_text("\u2033") == '"'

    def test_smart_double_quotes(self):
        assert _normalize_text("\u201chello\u201d") == '"hello"'

    def test_combined(self):
        raw = "  \ufb01nding\u2014the \u2019best\u2019  "
        assert _normalize_text(raw) == "finding-the 'best'"


class TestCleanTextForPdf:
    def test_empty(self):
        assert _clean_text_for_pdf("") == ""

    def test_none(self):
        assert _clean_text_for_pdf(None) is None

    def test_ligatures_fi_fl(self):
        assert "\ufb01" not in _clean_text_for_pdf("e\ufb01ect")
        assert "fi" in _clean_text_for_pdf("e\ufb01ect")
        assert "\ufb02" not in _clean_text_for_pdf("\ufb02uid")
        assert "fl" in _clean_text_for_pdf("\ufb02uid")

    def test_ligatures_ff_ffi_ffl(self):
        assert "ff" in _clean_text_for_pdf("\ufb00ect")
        assert "ffi" in _clean_text_for_pdf("\ufb03rm")
        assert "ffl" in _clean_text_for_pdf("\ufb04ected")

    def test_dashes(self):
        for ch in "\u2010\u2011\u2012\u2013\u2014\u2015":
            assert ch not in _clean_text_for_pdf(f"a{ch}b")

    def test_quotes(self):
        assert "\u2019" not in _clean_text_for_pdf("it\u2019s")
        assert "\u201c" not in _clean_text_for_pdf("\u201chi\u201d")
        assert "\u2018" not in _clean_text_for_pdf("\u2018x\u2019")

    def test_angle_quotes(self):
        assert "<" in _clean_text_for_pdf("\u2039")
        assert ">" in _clean_text_for_pdf("\u203a")

    def test_guillemets(self):
        assert "<<" in _clean_text_for_pdf("\u00ab")
        assert ">>" in _clean_text_for_pdf("\u00bb")

    def test_ellipsis(self):
        assert "..." in _clean_text_for_pdf("\u2026")

    def test_nbsp(self):
       
        assert _clean_text_for_pdf("\u00a0") == ""
       
        assert "hello world" in _clean_text_for_pdf("hello\u00a0world")

    def test_bullet(self):
        assert "-" in _clean_text_for_pdf("\u2022")

    def test_fraction_slash(self):
        assert "/" in _clean_text_for_pdf("1\u20442")

    def test_soft_hyphen(self):
        assert _clean_text_for_pdf("soft\u00adhyphen") == "softhyphen"

    def test_degree(self):
        assert " degrees " in _clean_text_for_pdf("37\u00b0C")

    def test_plus_minus(self):
        assert "+/-" in _clean_text_for_pdf("5\u00b13")

    def test_multiply(self):
        assert "x" in _clean_text_for_pdf("3\u00d72")

    def test_divide(self):
        assert "/" in _clean_text_for_pdf("6\u00f72")

    def test_minus_sign(self):
        assert "-" in _clean_text_for_pdf("\u22121")

    def test_leq(self):
        assert "<=" in _clean_text_for_pdf("x\u2264y")

    def test_geq(self):
        assert ">=" in _clean_text_for_pdf("x\u2265y")

    def test_neq(self):
        assert "!=" in _clean_text_for_pdf("x\u2260y")

    def test_per_mille(self):
        assert " per thousand" in _clean_text_for_pdf("\u2030")

    def test_per_ten_thousand(self):
        assert " per ten thousand" in _clean_text_for_pdf("\u2031")

    def test_superscript_digits(self):
        sup_map = {'\u2070': '0', '\u00b9': '1', '\u00b2': '2', '\u00b3': '3',
                   '\u2074': '4', '\u2075': '5', '\u2076': '6', '\u2077': '7',
                   '\u2078': '8', '\u2079': '9', '\u207b': '-', '\u207a': '+'}
        for k, v in sup_map.items():
            assert v in _clean_text_for_pdf(f"x{k}")

    def test_subscript_digits(self):
        sub_map = {'\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3',
                   '\u2084': '4', '\u2085': '5', '\u2086': '6', '\u2087': '7',
                   '\u2088': '8', '\u2089': '9', '\u208b': '-'}
        for k, v in sub_map.items():
            assert v in _clean_text_for_pdf(f"H{k}")

    def test_non_ascii_replaced_with_space(self):
        cleaned = _clean_text_for_pdf("hello\u0100world")
        assert "hello" in cleaned
        assert "world" in cleaned
        assert "\u0100" not in cleaned

    def test_ascii_preserved(self):
        text = "Hello World 123 !@#"
        assert _clean_text_for_pdf(text) == text


class TestFindNextNonOverlappingX:
    def test_no_overlap_returns_base(self):
        result = _find_next_non_overlapping_x([], 100, 200, 12, 12)
        assert result == 100

    def test_no_overlap_with_existing_rects(self):
        placed = [fitz.Rect(50, 188, 62, 200)]
        result = _find_next_non_overlapping_x(placed, 100, 200, 12, 12)
        assert result == 100

    def test_overlap_shifts_right(self):
        placed = [fitz.Rect(99, 188, 113, 200)]
        result = _find_next_non_overlapping_x(placed, 100, 200, 12, 12)
        assert result > 100

    def test_all_overlapping_returns_base(self):
        placed = [fitz.Rect(99 + offset, 188, 113 + offset, 200) for offset in range(0, 500, 5)]
        result = _find_next_non_overlapping_x(placed, 100, 200, 12, 12)
        assert result == 100


class TestWrapText:
    def test_short_text(self):
        assert _wrap_text("hello world", 85) == ["hello world"]

    def test_long_text_wraps(self):
        text = "word " * 30
        lines = _wrap_text(text, 85)
        for line in lines:
            assert len(line) <= 85

    def test_long_word_broken(self):
        text = "a" * 200
        lines = _wrap_text(text, 85)
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 85

    def test_single_word(self):
        assert _wrap_text("hello", 85) == ["hello"]

    def test_multiple_words_near_limit(self):
        text = "a " * 43   
        lines = _wrap_text(text.strip(), 85)
        assert len(lines) >= 1


class TestJsEscape:
    def test_backslash(self):
        assert js_escape("a\\b") == "a\\\\b"

    def test_backtick(self):
        assert js_escape("a`b") == "a\\`b"

    def test_dollar_brace(self):
        assert js_escape("${x}") == "\\${x}"

    def test_newline(self):
        assert js_escape("a\nb") == "a\\nb"

    def test_no_special_chars(self):
        assert js_escape("hello world") == "hello world"

    def test_combined(self):

        assert js_escape("\\`\n${") == "\\\\\\`\\n\\${"


# ======================================================================
# 2. PDF text-search functions
# ======================================================================


class TestFindTextAreas:
    def test_empty_text_returns_empty(self, dummy_pdf_page):
        assert find_text_areas(dummy_pdf_page, "") == []
        assert find_text_areas(dummy_pdf_page, "   ") == []

    def test_found_text_returns_rects(self, dummy_pdf_page):
        areas = find_text_areas(dummy_pdf_page, "randomly assigned")
        assert len(areas) > 0
        for a in areas:
            assert isinstance(a, fitz.Rect)

    def test_not_found_returns_empty(self, dummy_pdf_page):
        areas = find_text_areas(dummy_pdf_page, "xyzzy_not_in_pdf_12345")
        assert areas == []

    def test_header_filtering(self, dummy_pdf_page):
        areas = find_text_areas(dummy_pdf_page, "randomly assigned", header_height=10000)
        assert areas == []

    def test_short_text_found(self, dummy_pdf_page):
        areas = find_text_areas(dummy_pdf_page, "Drug X")
        assert len(areas) > 0

    def test_exact_line_found(self, dummy_pdf_page):
        areas = find_text_areas(dummy_pdf_page, "Abstract")
        assert len(areas) > 0


class TestExpandToLines:
    def test_expand_small_rect_to_line(self, dummy_pdf_page):
        areas = dummy_pdf_page.search_for("randomly assigned")
        if not areas:
            pytest.skip("Substring not found in dummy PDF")
        small_rect = areas[0]
        td = dummy_pdf_page.get_text("dict")
        expanded = _expand_to_lines(dummy_pdf_page, small_rect, header_height=0, td=td)
        assert expanded.width >= small_rect.width - 1

    def test_returns_original_on_failure(self, dummy_pdf_page):
        small_rect = fitz.Rect(1, 1, 2, 2)
        result = _expand_to_lines(dummy_pdf_page, small_rect, header_height=0)
        assert result == small_rect

    def test_header_height_filters_lines(self, dummy_pdf_page):
        areas = dummy_pdf_page.search_for("Abstract")
        if not areas:
            pytest.skip("Substring not found")
        small_rect = areas[0]
  
        result = _expand_to_lines(dummy_pdf_page, small_rect, header_height=800)
      
        assert result == small_rect


# ======================================================================
# 3. Highlight & anchor functions
# ======================================================================


class TestApplyHighlightAndGetAnchor:
    def test_found_text_returns_coords(self, dummy_pdf_page):
        td = dummy_pdf_page.get_text("dict")
        page_text_cache = _normalize_text(dummy_pdf_page.get_text("text"))
        sup_x, sup_y = _apply_highlight_and_get_anchor(
            dummy_pdf_page,
            "randomly assigned",
            header_height=0,
            highlight_color=(1.0, 0.9, 0.9),
            td=td,
            page_text_cache=page_text_cache,
        )
        assert sup_x is not None
        assert sup_y is not None

    def test_not_found_returns_none(self, dummy_pdf_page):
        td = dummy_pdf_page.get_text("dict")
        page_text_cache = _normalize_text(dummy_pdf_page.get_text("text"))
        sup_x, sup_y = _apply_highlight_and_get_anchor(
            dummy_pdf_page,
            "xyzzy_not_in_pdf_12345",
            header_height=0,
            highlight_color=(1.0, 0.9, 0.9),
            td=td,
            page_text_cache=page_text_cache,
        )
        assert sup_x is None
        assert sup_y is None

    def test_cache_miss_returns_none_early(self, dummy_pdf_page):
        sup_x, sup_y = _apply_highlight_and_get_anchor(
            dummy_pdf_page,
            "randomly assigned",
            header_height=0,
            highlight_color=(1.0, 0.9, 0.9),
            page_text_cache="zzzzzzzzzzzz",
        )
        assert sup_x is None
        assert sup_y is None

    def test_header_height_filters_result(self, dummy_pdf_page):
        td = dummy_pdf_page.get_text("dict")
        page_text_cache = _normalize_text(dummy_pdf_page.get_text("text"))

        sup_x, sup_y = _apply_highlight_and_get_anchor(
            dummy_pdf_page,
            "randomly assigned",
            header_height=10000,
            highlight_color=(1.0, 0.9, 0.9),
            td=td,
            page_text_cache=page_text_cache,
        )
        assert sup_x is None
        assert sup_y is None

    def test_none_cache_skips_prefix_check(self, dummy_pdf_page):
        td = dummy_pdf_page.get_text("dict")
 
        sup_x, sup_y = _apply_highlight_and_get_anchor(
            dummy_pdf_page,
            "randomly assigned",
            header_height=0,
            highlight_color=(1.0, 0.9, 0.9),
            td=td,
            page_text_cache=None,
        )
        assert sup_x is not None


# ======================================================================
# 4. Highlighted-PDF generation
# ======================================================================


class TestCreateBiasHighlightedPdf:
    def test_creates_valid_pdf(self, dummy_pdf_bytes, sample_bias_annotations):
        result = create_bias_highlighted_pdf(dummy_pdf_bytes, sample_bias_annotations)
        assert isinstance(result, bytes)
        assert len(result) > 0
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) > 0
        doc.close()

    def test_empty_annotations(self, dummy_pdf_bytes):
        result = create_bias_highlighted_pdf(dummy_pdf_bytes, [])
        assert isinstance(result, bytes)
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) > 0
        doc.close()

    def test_header_note_is_present(self, dummy_pdf_bytes, sample_bias_annotations):
        result = create_bias_highlighted_pdf(dummy_pdf_bytes, sample_bias_annotations)
        doc = fitz.open(stream=result, filetype="pdf")
        page = doc[0]
        text = page.get_text("text")
        assert "Note:" in text or "optimal accuracy" in text
        doc.close()

    def test_annotations_with_missing_text_dont_crash(self, dummy_pdf_bytes):
        annotations = [
            {"type": "bias", "text": "", "bias_domain": "Random sequence generation"},
            {"type": "bias", "text": "some text", "bias_domain": ""},
            {"type": "bias", "text": "", "bias_domain": ""},
        ]
        result = create_bias_highlighted_pdf(dummy_pdf_bytes, annotations)
        assert isinstance(result, bytes)

    def test_annotation_with_nonexistent_text(self, dummy_pdf_bytes):
        annotations = [
            {"type": "bias", "text": "this text does not exist anywhere", "bias_domain": "Random sequence generation"},
        ]
        result = create_bias_highlighted_pdf(dummy_pdf_bytes, annotations)
        assert isinstance(result, bytes)


class TestCreatePicoHighlightedPdf:
    def test_creates_valid_pdf(self, dummy_pdf_bytes, sample_pico_annotations):
        result = create_pico_highlighted_pdf(dummy_pdf_bytes, sample_pico_annotations)
        assert isinstance(result, bytes)
        assert len(result) > 0
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) > 0
        doc.close()

    def test_empty_annotations(self, dummy_pdf_bytes):
        result = create_pico_highlighted_pdf(dummy_pdf_bytes, [])
        assert isinstance(result, bytes)
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) > 0
        doc.close()

    def test_legend_is_present(self, dummy_pdf_bytes, sample_pico_annotations):
        result = create_pico_highlighted_pdf(dummy_pdf_bytes, sample_pico_annotations)
        doc = fitz.open(stream=result, filetype="pdf")
        page = doc[0]
        text = page.get_text("text")
       
        assert "Population" in text or "Intervention" in text or "Outcomes" in text
        doc.close()


# ======================================================================
# 5. Evidence-PDF generation
# ======================================================================


class TestInsertEvidenceTextbox:
    def test_text_fits_on_page(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        y = 80
        page, y_out = _insert_evidence_textbox(
            page, doc, "Short test text.", y,
            page_width=595, margin_left=50, margin_right=50,
            bottom_margin=50, margin_top=50,
        )
        assert y_out > y
        doc.close()

    def test_long_text_creates_new_page(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        y = 800
        long_text = "Word " * 500
        page, y_out = _insert_evidence_textbox(
            page, doc, long_text, y,
            page_width=595, margin_left=50, margin_right=50,
            bottom_margin=50, margin_top=50,
        )
        assert len(doc) > 1
        doc.close()

    def test_y_advances_after_insert(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        y = 100
        page, y_out = _insert_evidence_textbox(
            page, doc, "Some text here.", y,
            page_width=595, margin_left=50, margin_right=50,
            bottom_margin=50, margin_top=50,
        )
        assert y_out > y
        doc.close()


class TestCreateBiasEvidencePdf:
    def test_creates_valid_pdf(self, sample_bias_results):
        result = create_bias_evidence_pdf(sample_bias_results, "test_trial.pdf")
        assert isinstance(result, bytes)
        assert len(result) > 0
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) > 0
        text = ""
        for page in doc:
            text += page.get_text("text")
        assert "Random sequence generation" in text
        doc.close()

    def test_empty_results(self):
        result = create_bias_evidence_pdf([], "empty.pdf")
        assert isinstance(result, bytes)
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) > 0
        doc.close()

    def test_no_evidence_texts(self):
        results = [{"domain": "Test Domain", "judgement": "low", "text": []}]
        result = create_bias_evidence_pdf(results, "test.pdf")
        assert isinstance(result, bytes)
        doc = fitz.open(stream=result, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        assert "No evidence sentences extracted" in text
        doc.close()

    def test_high_unclear_judgement(self):
        results = [{"domain": "Test Domain", "judgement": "high", "text": ["some evidence"]}]
        result = create_bias_evidence_pdf(results, "test.pdf")
        assert isinstance(result, bytes)
        doc = fitz.open(stream=result, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        assert "High/Unclear" in text
        doc.close()

    def test_filename_in_output(self):
        results = [{"domain": "D", "judgement": "low", "text": ["evidence"]}]
        result = create_bias_evidence_pdf(results, "my_study.pdf")
        doc = fitz.open(stream=result, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        assert "my_study.pdf" in text
        doc.close()


class TestCreatePicoEvidencePdf:
    def test_creates_valid_pdf(self, sample_pico_results):
        result = create_pico_evidence_pdf(sample_pico_results, "test_trial.pdf")
        assert isinstance(result, bytes)
        assert len(result) > 0
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) > 0
        text = ""
        for page in doc:
            text += page.get_text("text")
        assert "Population" in text
        assert "Intervention" in text
        assert "Outcomes" in text
        doc.close()

    def test_empty_results(self):
        result = create_pico_evidence_pdf([], "empty.pdf")
        assert isinstance(result, bytes)
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) > 0
        doc.close()

    def test_missing_domain_gets_no_texts(self):
        results = [
            {"domain": "Population", "text": ["500 patients"]},
      
            {"domain": "Outcomes", "text": ["symptom score"]},
        ]
        result = create_pico_evidence_pdf(results, "test.pdf")
        assert isinstance(result, bytes)

    def test_filename_in_output(self):
        results = [{"domain": "Population", "text": ["evidence"]}]
        result = create_pico_evidence_pdf(results, "my_study.pdf")
        doc = fitz.open(stream=result, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        assert "my_study.pdf" in text
        doc.close()


# ======================================================================
# 6. Export functions
# ======================================================================


class TestExportToJson:
    def test_valid_output(self, sample_analysis_results):
        output = export_to_json(sample_analysis_results)
        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["filename"] == "test_trial.pdf"
        assert data[0]["rct"]["is_rct"] is True
        assert "pico" in data[0]
        assert "bias" in data[0]
        assert "timestamp" in data[0]

    def test_empty_results(self):
        output = export_to_json([])
        data = json.loads(output)
        assert data == []

    def test_multiple_results(self, sample_analysis_results):
        results = sample_analysis_results + sample_analysis_results
        output = export_to_json(results)
        data = json.loads(output)
        assert len(data) == 2


class TestExportToCsv:
    def test_valid_output(self, sample_analysis_results):
        output = export_to_csv(sample_analysis_results)
        df = pd.read_csv(io.StringIO(output))
        assert len(df) == 1
        assert df.iloc[0]["filename"] == "test_trial.pdf"
        assert "pico_population" in df.columns
        assert "bias_random_sequence_generation" in df.columns

    def test_empty_results(self):
        output = export_to_csv([])
        assert isinstance(output, str)
        assert output == "\n"

    def test_is_rct_column(self, sample_analysis_results):
        output = export_to_csv(sample_analysis_results)
        df = pd.read_csv(io.StringIO(output))
        assert df.iloc[0]["is_rct"] == True

    def test_rct_score_column(self, sample_analysis_results):
        output = export_to_csv(sample_analysis_results)
        df = pd.read_csv(io.StringIO(output))
        assert "rct_score" in df.columns
        assert "rct_probability" in df.columns


# ======================================================================
# 7. Constants
# ======================================================================


class TestConstants:
    def test_pico_colors_keys(self):
        assert set(PICO_COLORS.keys()) == {"Population", "Intervention", "Outcomes"}

    def test_pico_letters_keys(self):
        assert set(PICO_LETTERS.keys()) == {"Population", "Intervention", "Outcomes"}

    def test_bias_colors_keys(self):
        expected = {
            "Random sequence generation",
            "Allocation concealment",
            "Blinding of participants and personnel",
            "Blinding of outcome assessment",
            "Incomplete outcome data",
            "Selective reporting",
        }
        assert set(BIAS_COLORS.keys()) == expected

    def test_bias_letters_keys(self):
        assert set(BIAS_LETTERS.keys()) == set(BIAS_COLORS.keys())

    def test_color_values_are_tuples_of_three(self):
        for v in PICO_COLORS.values():
            assert isinstance(v, tuple)
            assert len(v) == 3
        for v in BIAS_COLORS.values():
            assert isinstance(v, tuple)
            assert len(v) == 3

    def test_letter_values_are_single_chars(self):
        for v in PICO_LETTERS.values():
            assert isinstance(v, str)
            assert len(v) == 1
        for v in BIAS_LETTERS.values():
            assert isinstance(v, str)
            assert len(v) == 1


# ======================================================================
# 8. Model-management functions (mocked)
# ======================================================================


class TestDownloadModels:
    def test_already_cached(self, tmp_path):
        pico_dir = tmp_path / "pico"
        pico_dir.mkdir(parents=True)
        (pico_dir / "P_model.npz").write_bytes(b"fake")

        with patch("rct_reviewer.app2.MODELS_DIR", tmp_path):
            result = download_models()
        assert result is True

    def test_already_cached_with_callbacks(self, tmp_path):
        pico_dir = tmp_path / "pico"
        pico_dir.mkdir(parents=True)
        (pico_dir / "P_model.npz").write_bytes(b"fake")

        mock_bar = MagicMock()
        mock_status = MagicMock()

        with patch("rct_reviewer.app2.MODELS_DIR", tmp_path):
            result = download_models(progress_bar=mock_bar, status_text=mock_status)

        assert result is True
        mock_bar.progress.assert_called()
        mock_status.success.assert_called()

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.HfApi")
    def test_successful_download(self, mock_api_cls, mock_download, tmp_path):
        mock_api = mock_api_cls.return_value
        mock_api.list_repo_files.return_value = ["model.npz"]

        mock_status = MagicMock()

        with patch("rct_reviewer.app2.MODELS_DIR", tmp_path):
            result = download_models(status_text=mock_status)

        assert result is True
        mock_download.assert_called()
        mock_status.success.assert_called()

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.HfApi")
    def test_api_failure(self, mock_api_cls, mock_download, tmp_path):
        mock_api = mock_api_cls.return_value
        mock_api.list_repo_files.side_effect = Exception("API error")

        mock_status = MagicMock()

        with patch("rct_reviewer.app2.MODELS_DIR", tmp_path):
            result = download_models(status_text=mock_status)

        assert result is False
        mock_status.error.assert_called()

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.HfApi")
    def test_download_retry_failure(self, mock_api_cls, mock_download, tmp_path):
        mock_api = mock_api_cls.return_value
        mock_api.list_repo_files.return_value = ["model.npz"]
        mock_download.side_effect = Exception("Network error")

        mock_status = MagicMock()

        with patch("rct_reviewer.app2.MODELS_DIR", tmp_path):
            with patch("rct_reviewer.app2.time.sleep"):
                result = download_models(status_text=mock_status)

        assert result is False

    @patch("huggingface_hub.HfApi")
    def test_empty_repo(self, mock_api_cls, tmp_path):
        mock_api = mock_api_cls.return_value
        mock_api.list_repo_files.return_value = []

        mock_status = MagicMock()

        with patch("rct_reviewer.app2.MODELS_DIR", tmp_path):
            result = download_models(status_text=mock_status)

        assert result is False
        mock_status.error.assert_called()

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.HfApi")
    def test_download_with_progress(self, mock_api_cls, mock_download, tmp_path):
        mock_api = mock_api_cls.return_value
        mock_api.list_repo_files.return_value = ["file1.npz", "file2.npz"]

        mock_bar = MagicMock()
        mock_status = MagicMock()

        with patch("rct_reviewer.app2.MODELS_DIR", tmp_path):
            result = download_models(progress_bar=mock_bar, status_text=mock_status)

        assert result is True
      
        assert mock_bar.progress.call_count >= 2

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.HfApi")
    def test_retry_succeeds_on_second_attempt(self, mock_api_cls, mock_download, tmp_path):
        mock_api = mock_api_cls.return_value
        mock_api.list_repo_files.return_value = ["model.npz"]


        mock_download.side_effect = [Exception("timeout"), None]

        mock_status = MagicMock()

        with patch("rct_reviewer.app2.MODELS_DIR", tmp_path):
            with patch("rct_reviewer.app2.time.sleep"):
                result = download_models(status_text=mock_status)

        assert result is True
        mock_status.warning.assert_called()


class TestLoadModelsWithProgress:
    @patch("rct_reviewer.app2.BiasRobot")
    @patch("rct_reviewer.app2.PICORobot")
    @patch("rct_reviewer.app2.RCTRobot")
    def test_successful_load(self, mock_rct_cls, mock_pico_cls, mock_bias_cls):
        mock_rct_cls.return_value = MagicMock(name="RCTModel")
        mock_pico_cls.return_value = MagicMock(name="PICOModel")
        mock_bias_cls.return_value = MagicMock(name="BiasModel")

        models = load_models_with_progress()

        assert "rct" in models
        assert "pico" in models
        assert "bias" in models
        mock_rct_cls.assert_called_once()
        mock_pico_cls.assert_called_once()
        mock_bias_cls.assert_called_once()

    @patch("rct_reviewer.app2.BiasRobot")
    @patch("rct_reviewer.app2.PICORobot")
    @patch("rct_reviewer.app2.RCTRobot")
    def test_with_progress_callbacks(self, mock_rct_cls, mock_pico_cls, mock_bias_cls):
        mock_rct_cls.return_value = MagicMock()
        mock_pico_cls.return_value = MagicMock()
        mock_bias_cls.return_value = MagicMock()

        mock_bar = MagicMock()
        mock_status = MagicMock()

        models = load_models_with_progress(progress_bar=mock_bar, status_text=mock_status)

        assert models is not None
        mock_bar.progress.assert_called()
        mock_status.info.assert_called()
        mock_status.success.assert_called()

    @patch("rct_reviewer.app2.BiasRobot")
    @patch("rct_reviewer.app2.PICORobot")
    @patch("rct_reviewer.app2.RCTRobot")
    def test_returns_three_models(self, mock_rct_cls, mock_pico_cls, mock_bias_cls):
        mock_rct_cls.return_value = MagicMock()
        mock_pico_cls.return_value = MagicMock()
        mock_bias_cls.return_value = MagicMock()

        models = load_models_with_progress()

        assert len(models) == 3


# ======================================================================
# 9. Integration-style: full pipeline with dummy PDF + mocked models
# ======================================================================


class TestFullPipeline:
    """End-to-end test using the dummy PDF and mocked ML models,
    verifying that the whole analysis→highlight→evidence→export chain
    runs without errors."""

    @patch("rct_reviewer.app2.BiasRobot")
    @patch("rct_reviewer.app2.PICORobot")
    @patch("rct_reviewer.app2.RCTRobot")
    def test_analysis_to_export(
        self, mock_rct_cls, mock_pico_cls, mock_bias_cls,
        dummy_pdf_bytes, sample_analysis_results,
    ):
        mock_rct = MagicMock()
        mock_rct.predict.return_value = {
            "is_rct": True, "score": 0.95, "probability": 0.97
        }
        mock_rct_cls.return_value = mock_rct

        mock_pico = MagicMock()
        mock_pico.annotate.return_value = [
            {"domain": "Population", "text": ["500 participants"]},
            {"domain": "Intervention", "text": ["Drug X"]},
            {"domain": "Outcomes", "text": ["symptom severity"]},
        ]
        mock_pico_cls.return_value = mock_pico

        mock_bias = MagicMock()
        mock_bias.annotate.return_value = [
            {"domain": "Random sequence generation", "judgement": "low", "text": ["randomly assigned"]},
            {"domain": "Allocation concealment", "judgement": "low", "text": ["concealed"]},
            {"domain": "Blinding of participants and personnel", "judgement": "low", "text": ["blinded"]},
            {"domain": "Blinding of outcome assessment", "judgement": "unclear", "text": ["also blinded"]},
            {"domain": "Incomplete outcome data", "judgement": "low", "text": ["ITT"]},
            {"domain": "Selective reporting", "judgement": "low", "text": ["pre-specified"]},
        ]
        mock_bias_cls.return_value = mock_bias


        models = load_models_with_progress()
        assert models is not None

  
        parsed = {
            "title": "Test Trial",
            "abstract": "Abstract text",
            "sentences": ["sentence 1"],
            "text": "full text",
        }
        rct_res = models["rct"].predict(parsed["title"], parsed["abstract"])
        assert rct_res["is_rct"] is True

        pico_res = models["pico"].annotate(parsed["sentences"])
        assert len(pico_res) == 3

        bias_res = models["bias"].annotate(parsed["sentences"], parsed["text"])
        assert len(bias_res) == 6


        bias_annotations = []
        for b in bias_res:
            for t in b.get("text", []):
                bias_annotations.append({"type": "bias", "text": t, "bias_domain": b["domain"]})

        bias_hl_pdf = create_bias_highlighted_pdf(dummy_pdf_bytes, bias_annotations)
        assert isinstance(bias_hl_pdf, bytes)
        assert len(bias_hl_pdf) > 0

        pico_annotations = []
        for p in pico_res:
            for t in p.get("text", []):
                pico_annotations.append({"type": p["domain"], "text": t})

        pico_hl_pdf = create_pico_highlighted_pdf(dummy_pdf_bytes, pico_annotations)
        assert isinstance(pico_hl_pdf, bytes)
        assert len(pico_hl_pdf) > 0


        bias_ev_pdf = create_bias_evidence_pdf(bias_res, "test_trial.pdf")
        assert isinstance(bias_ev_pdf, bytes)

        pico_ev_pdf = create_pico_evidence_pdf(pico_res, "test_trial.pdf")
        assert isinstance(pico_ev_pdf, bytes)


        result_entry = {
            "filename": "test_trial.pdf",
            "rct": rct_res,
            "pico": pico_res,
            "bias": bias_res,
        }
        json_out = export_to_json([result_entry])
        json.loads(json_out)

        csv_out = export_to_csv([result_entry])
        pd.read_csv(io.StringIO(csv_out))