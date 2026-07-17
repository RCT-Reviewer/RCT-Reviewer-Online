"""
conftest.py — Shared test fixtures and Streamlit mock.

The Streamlit mock MUST be installed at module level BEFORE any test file
imports from rct_reviewer.app2, because app2 calls st.set_page_config()
and st.markdown() at import time.
"""

import sys
import io
from unittest.mock import MagicMock

import pytest
import fitz


# ======================================================================
# Mock Streamlit BEFORE any test file imports app2
# ======================================================================
_mock_st = MagicMock()
_mock_st.cache_resource = lambda f: f 

sys.modules["streamlit"] = _mock_st
sys.modules["streamlit.components"] = MagicMock()
sys.modules["streamlit.components.v1"] = MagicMock()


# ======================================================================
# Dummy PDF text — realistic clinical trial language so text-search
# and highlight functions can find and annotate real matches.
# ======================================================================
DUMMY_PDF_LINES = [
    "A Randomized Controlled Trial of Drug X",
    "Abstract",
    "Background: Disease Y affects millions of people worldwide.",
    "Methods: We conducted a randomized double-blind placebo-controlled trial.",
    "Patients were randomly assigned to receive Drug X or placebo.",
    "Allocation was concealed using sealed opaque envelopes.",
    "Both participants and personnel were blinded to treatment.",
    "Outcome assessment was also blinded.",
    "Incomplete outcome data was addressed by intention-to-treat analysis.",
    "Results: A total of 500 participants were enrolled.",
    "The primary outcome showed significant improvement.",
    "Conclusions: Drug X is effective and safe for Disease Y.",
    "The primary outcome was the change in symptom severity score at 12 weeks.",
    "Secondary outcomes included quality of life measures and adverse events.",
]


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def dummy_pdf_bytes():
    """Create a small realistic dummy PDF with clinical-trial text.
    Text starts at y=80 so there is room above for header-height filtering."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    y = 80
    for line in DUMMY_PDF_LINES:
        page.insert_text(fitz.Point(50, y), line, fontsize=10, fontname="helv")
        y += 16

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def dummy_pdf_page(dummy_pdf_bytes):
    """Return the first page of the dummy PDF (for lower-level tests)."""
    doc = fitz.open(stream=dummy_pdf_bytes, filetype="pdf")
    page = doc[0]
    yield page
    doc.close()


@pytest.fixture
def sample_bias_annotations():
    """Bias annotations whose text exists in the dummy PDF."""
    return [
        {
            "type": "bias",
            "text": "randomly assigned to receive Drug X or placebo",
            "bias_domain": "Random sequence generation",
        },
        {
            "type": "bias",
            "text": "sealed opaque envelopes",
            "bias_domain": "Allocation concealment",
        },
        {
            "type": "bias",
            "text": "blinded to treatment",
            "bias_domain": "Blinding of participants and personnel",
        },
        {
            "type": "bias",
            "text": "blinded",
            "bias_domain": "Blinding of outcome assessment",
        },
        {
            "type": "bias",
            "text": "intention-to-treat analysis",
            "bias_domain": "Incomplete outcome data",
        },
    ]


@pytest.fixture
def sample_pico_annotations():
    """PICO annotations whose text exists in the dummy PDF."""
    return [
        {
            "type": "Population",
            "text": "Patients were randomly assigned to receive Drug X or placebo",
        },
        {
            "type": "Intervention",
            "text": "Drug X",
        },
        {
            "type": "Outcomes",
            "text": "change in symptom severity score at 12 weeks",
        },
    ]


@pytest.fixture
def sample_pico_results():
    """PICO results for evidence-PDF generation."""
    return [
        {
            "domain": "Population",
            "text": [
                "Patients were randomly assigned to receive Drug X or placebo.",
                "A total of 500 participants were enrolled.",
            ],
        },
        {
            "domain": "Intervention",
            "text": [
                "Drug X 50 mg or matching placebo once daily for 12 weeks.",
            ],
        },
        {
            "domain": "Outcomes",
            "text": [
                "The primary outcome was the change in symptom severity score at 12 weeks.",
                "Secondary outcomes included quality of life measures.",
            ],
        },
    ]


@pytest.fixture
def sample_bias_results():
    """Bias results for evidence-PDF generation."""
    return [
        {
            "domain": "Random sequence generation",
            "judgement": "low",
            "text": ["Patients were randomly assigned to receive Drug X or placebo."],
        },
        {
            "domain": "Allocation concealment",
            "judgement": "low",
            "text": ["Allocation was concealed using sealed opaque envelopes."],
        },
        {
            "domain": "Blinding of participants and personnel",
            "judgement": "low",
            "text": ["Both participants and personnel were blinded to treatment."],
        },
        {
            "domain": "Blinding of outcome assessment",
            "judgement": "unclear",
            "text": ["Outcome assessment was also blinded."],
        },
        {
            "domain": "Incomplete outcome data",
            "judgement": "low",
            "text": ["Incomplete outcome data was addressed by intention-to-treat analysis."],
        },
        {
            "domain": "Selective reporting",
            "judgement": "low",
            "text": ["Selective reporting was minimized by pre-specifying all outcomes."],
        },
    ]


@pytest.fixture
def sample_analysis_results():
    """Full analysis results for export-function testing."""
    return [
        {
            "filename": "test_trial.pdf",
            "rct": {
                "is_rct": True,
                "score": 0.95,
                "probability": 0.97,
            },
            "pico": [
                {"domain": "Population", "text": ["500 participants with Disease Y"]},
                {"domain": "Intervention", "text": ["Drug X 50 mg daily"]},
                {"domain": "Outcomes", "text": ["symptom severity score at 12 weeks"]},
            ],
            "bias": [
                {"domain": "Random sequence generation", "judgement": "low", "text": ["randomly assigned"]},
                {"domain": "Allocation concealment", "judgement": "low", "text": ["sealed envelopes"]},
                {"domain": "Blinding of participants and personnel", "judgement": "low", "text": ["blinded"]},
                {"domain": "Blinding of outcome assessment", "judgement": "unclear", "text": ["also blinded"]},
                {"domain": "Incomplete outcome data", "judgement": "low", "text": ["intention-to-treat"]},
                {"domain": "Selective reporting", "judgement": "low", "text": ["pre-specifying outcomes"]},
            ],
        }
    ]