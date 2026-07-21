# Author:
#   Vihaan Sahu <pteroisvolitans12@gmail.com>

# This .py file downloads models from Hugging Face hub. This is the default recommended mode which is hosted online.



# The other .py files (app.py , app1.py) are developer run and so, may not have any latest functions / features which this default version has.


import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import streamlit.components.v1 as components
import logging
import base64
import pandas as pd
import fitz
import io
import numpy as np
import re
from datetime import datetime


st.set_page_config(
    page_title="RCT-Reviewer",
    layout="wide",
    page_icon="assets/favicon.ico",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    button[title="Toggle sidebar"] {display: none;}

    .main .block-container {
        padding-bottom: 120px;
    }

    .stMarkdown, .stText, .streamlit-expanderContent {
        font-size: 1.05rem; 
    }

    .streamlit-expanderHeader {
        font-size: 1rem !important;
    }

    .fixed-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        border-top: 1px solid #e6e6e6;
        padding: 15px 20px;
        z-index: 999;
        text-align: center;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        font-size: 0.9em;
        color: #555;
    }

    .fixed-footer .footer-text {
        font-weight: normal;
    }

    .fixed-footer a {
        text-decoration: none;
        color: #dd0050; 
        font-weight: 600;
    }

    .fixed-footer a:hover {
        text-decoration: underline;
    }

    .citation-box {
        border-left: 4px solid #4157a5;
        background-color: #e9ecef;
        padding: 1rem;
        margin: 1rem 0;
        color: #000 !important;
    }
</style>
""", unsafe_allow_html=True)

HF_REPO_ID = "Aurumz/RCT-Reviewer"
MODELS_DIR = Path.home() / ".cache" / "rct_reviewer" / "models"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def _format_eta(seconds):
    if seconds < 0 or seconds == float('inf') or seconds != seconds:
        return "calculating..."
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"~{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"~{minutes}m {secs}s"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"~{hours}h {mins}m"

def download_models(progress_bar=None, status_text=None):
    check_file = MODELS_DIR / "pico" / "P_model.npz"

    if check_file.exists():
        log.info("Models already exist in cache.")
        if progress_bar is not None:
            progress_bar.progress(1.0, text="Models already cached locally - ready!")
        if status_text is not None:
            status_text.success("Models already downloaded and cached.")
        return True

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        if status_text is not None:
            status_text.error(" `huggingface_hub` library not found. Please add it to requirements.txt.")
        return False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if status_text is not None:
        status_text.info("Models not found locally. Downloading from Hugging Face Hub (One-time setup)...")

    api = HfApi(token=os.getenv("HF_TOKEN"))
    try:
        files = api.list_repo_files(HF_REPO_ID, repo_type="model")
    except Exception as e:
        if status_text is not None:
            status_text.error(f" Failed to list repository files: {e}")
        return False

    total_files = len(files)
    if total_files == 0:
        if status_text is not None:
            status_text.error(" No files found in the repository.")
        return False

    max_retries = 3
    start_time = time.time()
    files_completed = 0

    for idx, fname in enumerate(files):
        retry_count = 0
        file_success = False

        while retry_count < max_retries:
            try:
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=fname,
                    repo_type="model",
                    local_dir=MODELS_DIR,
                    token=os.getenv("HF_TOKEN")
                )
                file_success = True
                break
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    if status_text is not None:
                        status_text.warning(f"⚠️ Download attempt {retry_count} for `{fname}` failed: {str(e)[:80]}... Retrying in 3s...")
                    time.sleep(3)
                else:
                    if status_text is not None:
                        status_text.error(f" Failed to download `{fname}` after {max_retries} attempts: {e}")
                    return False

        if not file_success:
            return False

        files_completed = idx + 1
        elapsed = time.time() - start_time
        files_remaining = total_files - files_completed

        if files_completed > 0 and files_remaining > 0:
            avg_time_per_file = elapsed / files_completed
            eta_seconds = avg_time_per_file * files_remaining
            eta_str = _format_eta(eta_seconds)
        elif files_remaining == 0:
            eta_str = "0s"
        else:
            eta_str = "calculating..."

        progress_pct = files_completed / total_files

        if progress_bar is not None:
            progress_bar.progress(
                progress_pct,
                text=f"Downloading models: {files_completed}/{total_files} files ({progress_pct*100:.1f}%) - Time remaining: {eta_str}"
            )

        if status_text is not None:
            status_text.info(f"Downloaded `{fname}` ({files_completed}/{total_files}) - ETA: {eta_str}")

    if progress_bar is not None:
        progress_bar.progress(1.0, text="All model files downloaded successfully! (100%)")
    if status_text is not None:
        status_text.success(f"Models downloaded successfully to: {MODELS_DIR}! Please do not clear this Streamlit's cache.")

    log.info("All model files downloaded successfully.")
    return True


import rct_reviewer
rct_reviewer.DATA_ROOT = MODELS_DIR

from rct_reviewer.config import settings
settings.use_joblib = True

from rct_reviewer.core.pdf_parser import PDFParser
from rct_reviewer.core.models import DocumentAnalysis
from rct_reviewer.ml.rct_robot import RCTRobot
from rct_reviewer.ml.pico_robot import PICORobot
from rct_reviewer.ml.bias_robot import BiasRobot


@st.cache_resource
def _load_rct_model():
    return RCTRobot()

@st.cache_resource
def _load_pico_model():
    return PICORobot()

@st.cache_resource
def _load_bias_model():
    return BiasRobot()

def load_models_with_progress(progress_bar=None, status_text=None):
    loaders = [
        ("RCT", _load_rct_model),
        ("PICO", _load_pico_model),
        ("Bias", _load_bias_model),
    ]
    total = len(loaders)
    models = {}
    start_time = time.time()

    for i, (name, fn) in enumerate(loaders):
        completed = i
        remaining = total - completed

        if completed > 0 and remaining > 0:
            elapsed = time.time() - start_time
            avg = elapsed / completed
            eta_seconds = avg * remaining
            eta_str = _format_eta(eta_seconds)
        elif remaining == 0:
            eta_str = "0s"
        else:
            eta_str = "calculating..."

        pct = completed / total

        if progress_bar is not None:
            progress_bar.progress(
                pct,
                text=f"Loading ML models: {name} ({completed}/{total} loaded) - Time remaining: {eta_str}"
            )
        if status_text is not None:
            status_text.info(f"Loading {name} model from cache... ({completed}/{total} done) - ETA: {eta_str}")

        models[name.lower()] = fn()

    if progress_bar is not None:
        progress_bar.progress(1.0, text="All ML models loaded successfully! (100%)")
    if status_text is not None:
        status_text.success("All models are fully loaded and ready.")

    return models


@st.cache_resource
def get_parser():
    return PDFParser()


PICO_COLORS = {
    "Population": (1.0, 0.76, 0.03),
    "Intervention": (1.0, 0.88, 0.0),
    "Outcomes": (0.93, 0.65, 0.0),
}

PICO_LETTERS = {
    "Population": "P",
    "Intervention": "I",
    "Outcomes": "O",
}

BIAS_COLORS = {
    "Random sequence generation": (1.0, 0.6, 0.6),
    "Allocation concealment": (1.0, 0.3, 0.3),
    "Blinding of participants and personnel": (0.86, 0.08, 0.24),
    "Blinding of outcome assessment": (0.8, 0.4, 0.0),
    "Incomplete outcome data": (0.8, 0.2, 0.2),
    "Selective reporting": (0.5, 0.0, 0.0),
}

BIAS_LETTERS = {
    "Random sequence generation": "R",
    "Allocation concealment": "A",
    "Blinding of participants and personnel": "B",
    "Blinding of outcome assessment": "O",
    "Incomplete outcome data": "I",
    "Selective reporting": "S",
}


def _normalize_text(t):
    """Normalize text for PDF search: whitespace, ligatures, dashes, quotes."""
    t = ' '.join(t.split())
    t = t.replace('\ufb01', 'fi').replace('\ufb02', 'fl').replace('\ufb00', 'ff').replace('\ufb03', 'ffi').replace('\ufb04', 'ffl')
    t = t.replace('\u2010', '-').replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
    t = t.replace('\u2032', "'").replace('\u2019', "'")
    t = t.replace('\u2033', '"').replace('\u201c', '"').replace('\u201d', '"')
    return t


def _clean_text_for_pdf(t):
    """Clean text for insertion into PDF with built-in fonts.
    Normalizes ligatures, special punctuation, and other Unicode that
    built-in fonts cannot render (would appear as dots)."""
    if not t:
        return t
    t = ' '.join(t.split())

    t = t.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    t = t.replace('\ufb00', 'ff').replace('\ufb03', 'ffi').replace('\ufb04', 'ffl')

    for ch in '\u2010\u2011\u2012\u2013\u2014\u2015':
        t = t.replace(ch, '-')

    t = t.replace('\u2018', "'").replace('\u2019', "'").replace('\u2032', "'")
    t = t.replace('\u201c', '"').replace('\u201d', '"').replace('\u2033', '"')
  
    t = t.replace('\u2039', '<').replace('\u203a', '>')
    t = t.replace('\u00ab', '<<').replace('\u00bb', '>>')

    t = t.replace('\u2026', '...')
    t = t.replace('\u00a0', ' ')     
    t = t.replace('\u2022', '-')     
    t = t.replace('\u2023', '-')     
    t = t.replace('\u2044', '/')     
    t = t.replace('\u00ad', '')      

    t = t.replace('\u00b0', ' degrees ')
    t = t.replace('\u00b1', '+/-')
    t = t.replace('\u00d7', 'x')      
    t = t.replace('\u00f7', '/')     
    t = t.replace('\u2212', '-')    
    t = t.replace('\u2264', '<=')    
    t = t.replace('\u2265', '>=')    
    t = t.replace('\u2260', '!=')     
    t = t.replace('\u2030', ' per thousand')
    t = t.replace('\u2031', ' per ten thousand')
  
    sup_map = {'\u2070': '0', '\u00b9': '1', '\u00b2': '2', '\u00b3': '3',
               '\u2074': '4', '\u2075': '5', '\u2076': '6', '\u2077': '7',
               '\u2078': '8', '\u2079': '9', '\u207b': '-', '\u207a': '+'}
    for k, v in sup_map.items():
        t = t.replace(k, v)

    sub_map = {'\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3',
               '\u2084': '4', '\u2085': '5', '\u2086': '6', '\u2087': '7',
               '\u2088': '8', '\u2089': '9', '\u208b': '-'}
    for k, v in sub_map.items():
        t = t.replace(k, v)

    result = []
    for ch in t:
        if ord(ch) <= 255:
            result.append(ch)
        else:
            result.append(' ')
    return ''.join(result)


def _expand_to_lines(page, small_rect, header_height, td=None):
    """Expand a small match rect to cover all lines of the containing paragraph/block."""
    try:
        if td is None:
            td = page.get_text("dict")
        matched_lines = []

        for block in td["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                line_rect = fitz.Rect(line["bbox"])
                if line_rect.y0 < header_height:
                    continue
                if not (small_rect.y1 < line_rect.y0 - 3 or small_rect.y0 > line_rect.y1 + 3):
                    if not (small_rect.x1 < line_rect.x0 - 100 or small_rect.x0 > line_rect.x1 + 100):
                        matched_lines.append(line_rect)

        if matched_lines:
            x0 = min(l.x0 for l in matched_lines)
            y0 = min(l.y0 for l in matched_lines)
            x1 = max(l.x1 for l in matched_lines)
            y1 = max(l.y1 for l in matched_lines)
            return fitz.Rect(x0, y0, x1, y1)
    except Exception as e:
        log.debug(f"expand_to_lines failed: {e}")

    return small_rect


def find_text_areas(page, text, header_height=0, td=None):
    """Robust multi-strategy text search that ensures full paragraph/line coverage."""
    if not text or not text.strip():
        return []

    def filter_header(areas):
        return [a for a in areas if a.y0 >= header_height]


    areas = filter_header(page.search_for(text))
    if areas:
        return areas

    normalized = _normalize_text(text)

    
    areas = filter_header(page.search_for(normalized))
    if areas:
        return areas


    portion_len = 25
    if len(normalized) > portion_len * 2:
        first_part = normalized[:portion_len]
        last_part = normalized[-portion_len:]

        first_areas = filter_header(page.search_for(first_part))
        last_areas = filter_header(page.search_for(last_part))

        if first_areas and last_areas:
            for fa in first_areas:
                fa_exp = _expand_to_lines(page, fa, header_height, td=td)
                for la in last_areas:
                    la_exp = _expand_to_lines(page, la, header_height, td=td)
                    if abs(fa_exp.y0 - la_exp.y0) < 100:
                        expanded = fitz.Rect(
                            min(fa_exp.x0, la_exp.x0),
                            min(fa_exp.y0, la_exp.y0),
                            max(fa_exp.x1, la_exp.x1),
                            max(fa_exp.y1, la_exp.y1)
                        )
                        return [expanded]


    for length in [35, 30, 25, 20, 15]:
        if len(normalized) >= length:
            substr = normalized[:length]
            areas = filter_header(page.search_for(substr))
            if areas:
                return [_expand_to_lines(page, a, header_height, td=td) for a in areas]

  
    sentences = re.split(r'(?<=[.!?])\s+', normalized)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) >= 15:
            areas = filter_header(page.search_for(sent))
            if areas:
                return [_expand_to_lines(page, a, header_height, td=td) for a in areas]


    if len(normalized) > 60:
        mid = len(normalized) // 4
        for length in [35, 30, 25, 20]:
            if mid + length <= len(normalized):
                substr = normalized[mid:mid + length]
                areas = filter_header(page.search_for(substr))
                if areas:
                    return [_expand_to_lines(page, a, header_height, td=td) for a in areas]


    if len(normalized) > 60:
        for length in [35, 30, 25, 20]:
            substr = normalized[-length:]
            areas = filter_header(page.search_for(substr))
            if areas:
                return [_expand_to_lines(page, a, header_height, td=td) for a in areas]

  
    if len(normalized) > 20:
        for i in range(0, len(normalized) - 20, 8):
            window = normalized[i:i + 20]
            areas = filter_header(page.search_for(window))
            if areas:
                return [_expand_to_lines(page, a, header_height, td=td) for a in areas]

    return []


def _find_next_non_overlapping_x(placed_rects, base_x, base_y, box_w, box_h):
    """Find the next available X coordinate so superscripts don't overlap."""
    test_rect = fitz.Rect(base_x - 1, base_y - box_h, base_x + box_w, base_y)
    
    if not any(test_rect.intersects(r) for r in placed_rects):
        return base_x
        

    for offset in range(5, 500, 5):
        new_x = base_x + offset
        test_rect = fitz.Rect(new_x - 1, base_y - box_h, new_x + box_w, base_y)
        if not any(test_rect.intersects(r) for r in placed_rects):
            return new_x
            
    return base_x 


def _wrap_text(text, max_chars_per_line=85):
    """Wrap text to fit within max_chars_per_line characters per line."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(word) > max_chars_per_line:
            if current_line:
                lines.append(current_line)
                current_line = ""
            for i in range(0, len(word), max_chars_per_line):
                lines.append(word[i:i + max_chars_per_line])
        elif len(current_line) + len(word) + 1 <= max_chars_per_line:
            current_line += (" " if current_line else "") + word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def render_evidence_item(number, text):
    """Render a single evidence item, detecting and formatting tables."""
    lines = text.strip().split('\n')
    non_empty_lines = [l for l in lines if l.strip()]

    is_pipe_table = len(non_empty_lines) >= 2 and all('|' in line for line in non_empty_lines)
    is_tab_table = len(non_empty_lines) >= 2 and all('\t' in line for line in non_empty_lines)

    if is_pipe_table or is_tab_table:
        st.markdown(f"**{number}.**")
        if is_tab_table:
            md_lines = []
            for i, line in enumerate(non_empty_lines):
                cells = [c.strip() for c in line.split('\t')]
                md_lines.append('| ' + ' | '.join(cells) + ' |')
                if i == 0:
                    md_lines.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')
            st.markdown('\n'.join(md_lines))
        else:
            st.markdown(text)
    else:
        st.info(f"{number}. {text}")


def _apply_highlight_and_get_anchor(page, text, header_height, highlight_color, td=None, page_text_cache=None):
    """Applies highlight using quads for perfect multi-line tracing, fallback to rects.
    Returns the top-left anchor point (sup_x, sup_y) for the superscript."""
    if page_text_cache is not None:
        norm_text = _normalize_text(text)
        prefixes = [w[:4] for w in norm_text.split() if len(w) > 3][:5]
        if prefixes and not any(p in page_text_cache for p in prefixes):
            return None, None

    normalized = _normalize_text(text)
    
   
    quads = page.search_for(normalized, quads=True)
    if quads:
       
        valid_quads = [q for q in quads if q[0].y >= header_height]
        if valid_quads:
            annot = page.add_highlight_annot(valid_quads)
            annot.set_colors(stroke=highlight_color)
            annot.update()
        
            return valid_quads[0][0].x, valid_quads[0][0].y


    areas = find_text_areas(page, text, header_height, td=td)
    if areas:
        for area in areas:
            annot = page.add_highlight_annot(area)
            annot.set_colors(stroke=highlight_color)
            annot.update()
        return areas[0].x0, areas[0].y0

    return None, None


def create_bias_highlighted_pdf(pdf_bytes, annotations):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    current_date = datetime.now().strftime("%Y-%m-%d")

    highlight_color = (1.0, 0.9, 0.9)

    bias_legend_items = [
        ("R", "Random Sequence Generation", BIAS_COLORS["Random sequence generation"]),
        ("A", "Allocation Concealment", BIAS_COLORS["Allocation concealment"]),
        ("B", "Blinding of Participants", BIAS_COLORS["Blinding of participants and personnel"]),
        ("O", "Blinding of Outcome Assessment", BIAS_COLORS["Blinding of outcome assessment"]),
        ("I", "Incomplete Outcome Data", BIAS_COLORS["Incomplete outcome data"]),
        ("S", "Selective Reporting", BIAS_COLORS["Selective reporting"]),
    ]

    header_note = "Note: For optimal accuracy, use this together with the separately extracted evidence PDFs. Highlighting may not work in some PDFs due to differences in text formatting."

    for page_num, page in enumerate(doc):
        rect = page.rect
        header_height = 72
        header_rect = fitz.Rect(0, 0, rect.width, header_height)
        page.draw_rect(header_rect, color=(1, 1, 1), fill=(1, 1, 1))

       

        legend_items = [("", "Text Highlight", highlight_color)] + bias_legend_items
        col_positions = [10, 200, 390]
        row_ys = [14, 28, 42]

        for idx, (letter, label, color) in enumerate(legend_items):
            lx = col_positions[idx % 3]
            ly = row_ys[idx // 3]
            box_rect = fitz.Rect(lx, ly - 7, lx + 16, ly + 7)
            page.draw_rect(box_rect, color=color, fill=color, width=0)
            if letter:
                page.insert_text(fitz.Point(lx + 4, ly + 3), letter, fontsize=8, color=(0, 0, 0), fontname="helv")
            label_fs = 7 if not letter else 6.5
            page.insert_text(fitz.Point(lx + 20, ly + 3), label, fontsize=label_fs, color=(0.2, 0.2, 0.2), fontname="helv")

        page.insert_text(fitz.Point(10, 58), header_note, fontsize=6.5, color=(0.55, 0.2, 0.2), fontname="helv")

        page.draw_line(fitz.Point(0, header_height - 7), fitz.Point(rect.width, header_height - 7), color=(0.6, 0.6, 0.6), width=1.5)

        legend_dest_point = fitz.Point(rect.width / 2, (12 + 40) / 2)


        footer_text = f"Risk of Bias / RCT-Reviewer on {current_date}."
        footer_fontsize = 8.5
        text_width = len(footer_text) * footer_fontsize * 0.5
        logo_w = 18
        logo_h = 18
        logo_x = rect.width - 25 - text_width - logo_w - 5
        logo_y_pos = rect.height - 22
        try:
            page.insert_image(fitz.Rect(logo_x, logo_y_pos - logo_h + 4, logo_x + logo_w, logo_y_pos + 4), filename="assets/main_logo_zoomed.png")
        except Exception:
            pass
        page.insert_text(fitz.Point(logo_x + logo_w + 5, logo_y_pos), _clean_text_for_pdf(footer_text), fontsize=footer_fontsize, color=(0.5, 0.5, 0.5), fontname="helv")

        bias_annotations = [a for a in annotations if a.get("type") == "bias"]
        placed_superscripts = []
        page_text_cache = _normalize_text(page.get_text("text"))
        td = page.get_text("dict")

        for ann in bias_annotations:
            text = ann.get("text", "")
            bias_domain = ann.get("bias_domain", "")
            if not text or not bias_domain:
                continue

            color = BIAS_COLORS.get(bias_domain, (1.0, 0.3, 0.3))
            letter = BIAS_LETTERS.get(bias_domain, "?")

            try:
                sup_x, sup_y = _apply_highlight_and_get_anchor(page, text, header_height, highlight_color, td=td, page_text_cache=page_text_cache)
                
                if sup_x is not None and sup_y is not None:
                    
                    box_w = 12
                    box_h = 12
                    final_x = _find_next_non_overlapping_x(placed_superscripts, sup_x, sup_y, box_w, box_h)
                    box_rect = fitz.Rect(final_x - 1, sup_y - box_h, final_x + box_w, sup_y)
                    
                    page.draw_rect(box_rect, color=None, fill=color, width=0) 
                    page.insert_text(fitz.Point(final_x + 2, sup_y - 2), letter, fontsize=10, color=(0, 0, 0), fontname="helv") 
                    
                    placed_superscripts.append(box_rect)

                    link_dict = {"kind": 1, "page": page_num, "to": legend_dest_point, "zoom": 0}
                    link_annot = page.add_link_annot(box_rect, link_dict)
                    link_annot.update()
            except Exception as e:
                log.debug(f"Could not annotate bias text: {e}")

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def create_pico_highlighted_pdf(pdf_bytes, annotations):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    current_date = datetime.now().strftime("%Y-%m-%d")
  
    highlight_color = (1.0, 1.0, 0.6) 

    header_note = "Note: For optimal accuracy, use this together with the separately extracted evidence PDFs. Highlighting may not work in some PDFs due to differences in text formatting."

    for page_num, page in enumerate(doc):
        rect = page.rect
        header_height = 62
        header_rect = fitz.Rect(0, 0, rect.width, header_height)
        page.draw_rect(header_rect, color=(1, 1, 1), fill=(1, 1, 1))

        

        pico_legend_items = [
            ("P", "Population", (1.0, 0.76, 0.03)),
            ("I", "Intervention", (1.0, 0.88, 0.0)),
            ("O", "Outcomes", (0.93, 0.65, 0.0)),
        ]

        legend_x = 10
        legend_y = 15

    
        hl_box = fitz.Rect(legend_x, legend_y - 8, legend_x + 18, legend_y + 8)
        page.draw_rect(hl_box, color=highlight_color, fill=highlight_color, width=0)
        page.insert_text(fitz.Point(legend_x + 22, legend_y + 4), "Text Highlight", fontsize=7, color=(0.2, 0.2, 0.2), fontname="helv")
        legend_x += 130

        for letter, label, color in pico_legend_items:
            box_rect = fitz.Rect(legend_x, legend_y - 8, legend_x + 18, legend_y + 8)
            page.draw_rect(box_rect, color=color, fill=color, width=0)
            page.insert_text(fitz.Point(legend_x + 5, legend_y + 4), letter, fontsize=10, color=(0, 0, 0), fontname="helv")
            page.insert_text(fitz.Point(legend_x + 22, legend_y + 3), label, fontsize=7, color=(0.2, 0.2, 0.2), fontname="helv")
            legend_x += 130

        page.insert_text(fitz.Point(10, 35), header_note, fontsize=6.5, color=(0.55, 0.2, 0.2), fontname="helv")

        page.draw_line(fitz.Point(0, header_height - 16), fitz.Point(rect.width, header_height - 16), color=(0.6, 0.6, 0.6), width=1.5)

        legend_dest_point = fitz.Point(rect.width / 2, legend_y)

   
        footer_text = f"RCT-Reviewer on {current_date}."
        footer_fontsize = 8.5
        text_width = len(footer_text) * footer_fontsize * 0.5
        logo_w = 18
        logo_h = 18
        logo_x = rect.width - 25 - text_width - logo_w - 5
        logo_y_pos = rect.height - 22
        try:
            page.insert_image(fitz.Rect(logo_x, logo_y_pos - logo_h + 4, logo_x + logo_w, logo_y_pos + 4), filename="assets/main_logo_zoomed.png")
        except Exception:
            pass
        page.insert_text(fitz.Point(logo_x + logo_w + 5, logo_y_pos), _clean_text_for_pdf(footer_text), fontsize=footer_fontsize, color=(0.5, 0.5, 0.5), fontname="helv")

        pico_annotations = [a for a in annotations if a.get("type") in ["Population", "Intervention", "Outcomes"]]
        placed_superscripts = []
        page_text_cache = _normalize_text(page.get_text("text"))
        td = page.get_text("dict")

        for ann in pico_annotations:
            text = ann.get("text", "")
            ann_type = ann.get("type", "")
            if not text:
                continue

            color = PICO_COLORS.get(ann_type, (1.0, 0.76, 0.03))
            letter = PICO_LETTERS.get(ann_type, "P")

            try:
                sup_x, sup_y = _apply_highlight_and_get_anchor(page, text, header_height, highlight_color, td=td, page_text_cache=page_text_cache)
                
                if sup_x is not None and sup_y is not None:
           
                    box_w = 12
                    box_h = 12
                    final_x = _find_next_non_overlapping_x(placed_superscripts, sup_x, sup_y, box_w, box_h)
                    box_rect = fitz.Rect(final_x - 1, sup_y - box_h, final_x + box_w, sup_y)
                    
                    page.draw_rect(box_rect, color=None, fill=color, width=0) 
                    page.insert_text(fitz.Point(final_x + 2, sup_y - 2), letter, fontsize=10, color=(0, 0, 0), fontname="helv") 
                    
                    placed_superscripts.append(box_rect)

                    link_dict = {"kind": 1, "page": page_num, "to": legend_dest_point, "zoom": 0}
                    link_annot = page.add_link_annot(box_rect, link_dict)
                    link_annot.update()

            except Exception as e:
                log.debug(f"Could not annotate PICO text: {e}")

    buf = io.BytesIO()
    doc.save(buf, deflate=False, garbage=0)
    doc.close()
    return buf.getvalue()


def _insert_evidence_textbox(page, doc, text, y, page_width, margin_left, margin_right, bottom_margin, margin_top, fontsize=9, color=(0.15, 0.15, 0.15)):
    """Insert text into a PDF page using insert_textbox with proper page-break handling.
    
    insert_textbox returns:
      - positive value = unused height remaining in the rect (all text fit)
      - negative value = overflow height (text did NOT fit, magnitude = how much extra was needed)
    
    We compute text_used = rect_height - rc when rc >= 0 to advance y correctly.
    """
    page_height = page.rect.height
    remaining = text

    while remaining:
     
        if y > page_height - bottom_margin - 30:
            page = doc.new_page(width=page_width, height=page_height)
            y = margin_top

        text_rect = fitz.Rect(margin_left, y, page_width - margin_right, page_height - bottom_margin)
        rect_height = (page_height - bottom_margin) - y

        rc = page.insert_textbox(text_rect, remaining, fontsize=fontsize, color=color, fontname="helv")

        if rc >= 0:
       
            text_used = rect_height - rc
            y += text_used + 4
            remaining = ""
        else:

            text_used = rect_height + rc
            if text_used > 0:
                y += text_used + 4
            else:
               
                y = page_height - bottom_margin + 5

       
            total_needed = rect_height + abs(rc)
            if total_needed > 0 and len(remaining) > 0:
                ratio = max(rect_height, 1) / total_needed
                cut = max(int(len(remaining) * ratio) - 15, 1)
                cut = min(cut, len(remaining) - 1)
                
                while cut < len(remaining) - 1 and remaining[cut] != ' ':
                    cut += 1
                remaining = remaining[cut:].lstrip()
            else:
                remaining = ""

    return page, y


def create_bias_evidence_pdf(bias_results, filename):
    """Create a standalone PDF with all Risk of Bias evidence sentences, domain-wise and numbered.
    Uses insert_textbox for full-width left-to-right text and _clean_text_for_pdf to prevent
    missing characters (e.g. ligature 'fi' showing as dots)."""
    doc = fitz.open()
    current_date = datetime.now().strftime("%Y-%m-%d")

    page_width = 595
    page_height = 842
    margin_top = 50
    margin_left = 50
    margin_right = 50
    bottom_margin = 50

    note_text = ""

    page = doc.new_page(width=page_width, height=page_height)
    y = margin_top


    page.insert_text(fitz.Point(margin_left, y), _clean_text_for_pdf("RCT-Reviewer"), fontsize=20, color=(0.15, 0.15, 0.15), fontname="helv")
    try:
        page.insert_image(fitz.Rect(page_width - margin_right - 35, y - 20, page_width - margin_right, y + 10), filename="assets/main_logo_zoomed.png")
    except Exception:
        pass
    y += 22
    page.insert_text(fitz.Point(margin_left, y), _clean_text_for_pdf("Extracted Risk of Bias Evidence Sentences"), fontsize=13, color=(0.35, 0.35, 0.35), fontname="helv")
    y += 16
    page.insert_text(fitz.Point(margin_left, y), _clean_text_for_pdf(f"Generated on: {current_date}  |  Source: {filename}"), fontsize=9, color=(0.5, 0.5, 0.5), fontname="helv")
    y += 14
    page.draw_line(fitz.Point(margin_left, y), fitz.Point(page_width - margin_right, y), color=(0.75, 0.75, 0.75), width=0.5)
    y += 14


    note_rect_height = 30
    note_rect = fitz.Rect(margin_left, y, page_width - margin_right, y + note_rect_height)
    rc = page.insert_textbox(note_rect, _clean_text_for_pdf(note_text), fontsize=7.5, color=(0.55, 0.2, 0.2), fontname="helv")

    if rc >= 0:
        y += (note_rect_height - rc) + 12
    else:
        y += note_rect_height + 12


    for b in bias_results:
        domain = b.get('domain', 'Unknown Domain')
        texts = b.get('text', [])

        if y > page_height - 120:
            page = doc.new_page(width=page_width, height=page_height)
            y = margin_top


        page.insert_text(fitz.Point(margin_left, y), _clean_text_for_pdf(domain), fontsize=11, color=(0.2, 0.2, 0.55), fontname="helv")
        y += 4
        page.draw_line(fitz.Point(margin_left, y), fitz.Point(margin_left + 180, y), color=(0.4, 0.4, 0.55), width=0.5)
        y += 12


        judgement = b.get('judgement', 'N/A')
        display_j = 'Low' if judgement == 'low' else 'High/Unclear'
        page.insert_text(fitz.Point(margin_left, y), _clean_text_for_pdf(f"Judgement: {display_j}"), fontsize=9, color=(0.4, 0.4, 0.4), fontname="helv")
        y += 14

        if texts:
            for i, text in enumerate(texts, 1):
                clean_text = _clean_text_for_pdf(f"{i}. {text}")
                page, y = _insert_evidence_textbox(
                    page, doc, clean_text, y,
                    page_width, margin_left, margin_right, bottom_margin, margin_top,
                    fontsize=9, color=(0.15, 0.15, 0.15)
                )
                y += 16
        else:
            page.insert_text(fitz.Point(margin_left, y), _clean_text_for_pdf("No evidence sentences extracted."), fontsize=9, color=(0.6, 0.6, 0.6), fontname="helv")
            y += 13

        y += 14


    y = page_height - 35
    page.draw_line(fitz.Point(margin_left, y), fitz.Point(page_width - margin_right, y), color=(0.8, 0.8, 0.8), width=0.3)
    y += 12

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def create_pico_evidence_pdf(pico_results, filename):
    """Create a standalone PDF with all PICO evidence sentences, domain-wise and numbered.
    Uses insert_textbox for full-width left-to-right text and _clean_text_for_pdf to prevent
    missing characters (e.g. ligature 'fi' showing as dots)."""
    doc = fitz.open()
    current_date = datetime.now().strftime("%Y-%m-%d")

    page_width = 595
    page_height = 842
    margin_top = 50
    margin_left = 50
    margin_right = 50
    bottom_margin = 50

    note_text = ""

    page = doc.new_page(width=page_width, height=page_height)
    y = margin_top

    page.insert_text(fitz.Point(margin_left, y), _clean_text_for_pdf("RCT-Reviewer"), fontsize=20, color=(0.15, 0.15, 0.15), fontname="helv")
    try:
        page.insert_image(fitz.Rect(page_width - margin_right - 35, y - 20, page_width - margin_right, y + 10), filename="assets/main_logo_zoomed.png")
    except Exception:
        pass
    y += 22
    page.insert_text(fitz.Point(margin_left, y), _clean_text_for_pdf("Extracted PICO Evidence Sentences"), fontsize=13, color=(0.35, 0.35, 0.35), fontname="helv")
    y += 16
    page.insert_text(fitz.Point(margin_left, y), _clean_text_for_pdf(f"Generated on: {current_date}  |  Source: {filename}"), fontsize=9, color=(0.5, 0.5, 0.5), fontname="helv")
    y += 14
    page.draw_line(fitz.Point(margin_left, y), fitz.Point(page_width - margin_right, y), color=(0.75, 0.75, 0.75), width=0.5)
    y += 14


    note_rect_height = 30
    note_rect = fitz.Rect(margin_left, y, page_width - margin_right, y + note_rect_height)
    rc = page.insert_textbox(note_rect, _clean_text_for_pdf(note_text), fontsize=7.5, color=(0.55, 0.2, 0.2), fontname="helv")
    if rc >= 0:
        y += (note_rect_height - rc) + 12
    else:
        y += note_rect_height + 12


    pico_order = ["Population", "Intervention", "Outcomes"]
    for pico_domain in pico_order:
        domain_data = next((p for p in pico_results if p['domain'] == pico_domain), None)
        texts = domain_data.get('text', []) if domain_data else []

        if y > page_height - 120:
            page = doc.new_page(width=page_width, height=page_height)
            y = margin_top


        page.insert_text(fitz.Point(margin_left, y), _clean_text_for_pdf(pico_domain), fontsize=11, color=(0.2, 0.2, 0.55), fontname="helv")
        y += 4
        page.draw_line(fitz.Point(margin_left, y), fitz.Point(margin_left + 140, y), color=(0.4, 0.4, 0.55), width=0.5)
        y += 12

        if texts:
            for i, text in enumerate(texts, 1):
                clean_text = _clean_text_for_pdf(f"{i}. {text}")
                page, y = _insert_evidence_textbox(
                    page, doc, clean_text, y,
                    page_width, margin_left, margin_right, bottom_margin, margin_top,
                    fontsize=9, color=(0.15, 0.15, 0.15)
                )
                y += 16
        else:
            page.insert_text(fitz.Point(margin_left, y), _clean_text_for_pdf("No evidence sentences extracted."), fontsize=9, color=(0.6, 0.6, 0.6), fontname="helv")
            y += 13

        y += 14

  
    y = page_height - 35
    page.draw_line(fitz.Point(margin_left, y), fitz.Point(page_width - margin_right, y), color=(0.8, 0.8, 0.8), width=0.3)
    y += 12

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()



def export_to_json(results):
    import json
    data = []
    for r in results:
        data.append({
            "filename": r["filename"], "rct": r["rct"], "pico": r["pico"], "bias": r["bias"],
            "timestamp": datetime.now().isoformat()
        })
    return json.dumps(data, indent=2, default=str)


def export_to_csv(results):
    rows = []
    for r in results:
        row = {
            "filename": r["filename"], "is_rct": r["rct"]["is_rct"],
            "rct_score": r["rct"]["score"], "rct_probability": r["rct"]["probability"],
        }
        for p in r.get("pico", []):
            row[f"pico_{p['domain'].lower()}"] = " | ".join(p.get("text", []))
        for b in r.get("bias", []):
            row[f"bias_{b['domain'].lower().replace(' ', '_')}"] = b.get("judgement", "N/A")
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def js_escape(text):
    return text.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${').replace('\n', '\\n')


def main():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("assets/banner.svg", width=730)

    st.markdown("---")

    st.markdown("""
    **RCT-Reviewer** is a modernized, standalone version of [RobotReviewer](https://github.com/ijmarshall/robotreviewer), designed as a third-party reference tool for Risk of Bias assessment. It builds upon RobotReviewer's original machine learning models trained on **12,808 randomized controlled trials (RCTs)**.
    """)

    with st.expander("Why use RCT-Reviewer?"):
        st.markdown("""
        RCT-Reviewer is designed as a Third-Party Tiebreaker Reference for systematic reviews. Standard guidelines require two independent human reviewers; when they disagree, this tool provides an instant, objective, and data-driven third opinion to resolve ties.

        <ul>
            <li><strong>Near-Human Accuracy</strong>: The system achieves <strong>71.0% accuracy</strong> for Risk of Bias judgments, performing within <strong>&lt;8% of human expert consensus</strong> (which stands at 78.3%) <a href="#ref-1">[1]</a>.</li>
            <li><strong>Highly Precise Extraction</strong>: In a randomized Cochrane user trial, the models demonstrated <strong>87% Precision</strong> and <strong>90% Recall</strong> for identifying the exact text snippets supporting the bias judgment <a href="#ref-2">[2]</a>.</li>
            <li><strong>Validated Acceptance</strong>: Real-world feasibility studies show that human reviewers accept the tool's judgments at a rate equal to that of their human peers (Risk Ratio 1.02) <a href="#ref-3">[3]</a>.</li>
            <li><strong>Rigorous Methodology</strong>: Developed by Marshall, Kuiper, and Wallace, the models were trained on <strong>12,808 clinical trial PDFs</strong> using "distant supervision" to ensure high-quality classification without prohibitive manual labeling costs <a href="#ref-1">[1]</a>,<a href="#ref-4">[4]</a>.</li>
        </ul>
        """, unsafe_allow_html=True)


    with st.expander("🔄 Differences from Original RobotReviewer"):
        st.markdown("""
        | Feature | Original RobotReviewer (2017) | RCT-Reviewer (2026) |
        | :--- | :--- | :--- |
        | **Compatibility** | Compatible with Python 3.6 (Not Compatible for 3.9+) | Modernized for Python 3.12 |
        | **PDF Parsing** | GROBID (Requires Java/Docker) | PyMuPDF (Native Python / Modern) |
        | **Task Queue** | Celery + RabbitMQ | Synchronous (Local execution) |
        | **Data Models** | MultiDict | Pydantic |
        | **ML Core** | SVM / CNN | Same Weights (SVM prioritized) |
        | **Underlying ML Research** | Original ML models trained on 12,808 RCT PDFs | Preserves the same trained ML models and weights |
        | **Risk of Bias Accuracy** | ~71.0% agreement accuracy vs expert consensus | ~71.0% agreement accuracy vs expert consensus (Same SVM weights) |
        | **Supporting Text Precision** | ~87% precision for rationale extraction | ~87% precision for rationale extraction (Same extraction models) |
        | **Supporting Text Recall** | ~90% recall | ~90% recall (Same extraction models) |
        | **Model Storage** | Pickle / HDF5 / NPZ | Joblib / NPZ / legacy compatibility modes |
        | **Expected Accuracy Difference After CNN Removal** | Baseline reference | Estimated negligible reduction (~0–2%) |
        | **Interface** | Flask + React | Streamlit (Pure Python) |
        | **Deployment** | Docker Compose | Local Streamlit Run |
        | **Core Purpose** | Automated Risk of Bias assessment for RCTs | Modernized standalone implementation for automated Risk of Bias assessment |

        *For more information on Architecture, please visit the <a href="https://github.com/aurumz-rgb/RCT-Reviewer" target="_blank">GitHub Repository</a>.*
        """, unsafe_allow_html=True)

    if "models_ready" not in st.session_state:
        st.session_state.models_ready = False

    if not st.session_state.models_ready:
        overall_progress = st.progress(0.0, text="Initializing model setup...")
        overall_status = st.empty()

        overall_status.info("Phase 1/2: Checking and downloading models from Hugging Face Hub...")

        class _DownloadProgressProxy:
            def progress(self, pct, text=None):
                mapped = pct * 0.8
                overall_progress.progress(
                    mapped,
                    text=f"[ONE TIME DOWNLOAD] {text}" if text else f"[ONE TIME DOWNLOAD] {pct*100:.1f}%"
                )

        class _DownloadStatusProxy:
            def info(self, msg):
                overall_status.info(msg)
            def success(self, msg):
                overall_status.success(msg)
            def warning(self, msg):
                overall_status.warning(msg)
            def error(self, msg):
                overall_status.error(msg)

        success = download_models(
            progress_bar=_DownloadProgressProxy(),
            status_text=_DownloadStatusProxy()
        )

        if not success:
            overall_progress.empty()
            overall_status.error("Model download failed. Please check logs.")
            st.stop()

        overall_status.info("Phase 2/2: Loading ML models from cache into memory...")

        class _LoadProgressProxy:
            def progress(self, pct, text=None):
                mapped = 0.8 + pct * 0.2
                overall_progress.progress(
                    mapped,
                    text=f"[Load] {text}" if text else f"[Load] {pct*100:.1f}%"
                )

        models = load_models_with_progress(
            progress_bar=_LoadProgressProxy(),
            status_text=_DownloadStatusProxy()
        )
        parser = get_parser()

        overall_progress.progress(1.0, text="All models fully downloaded and loaded — ready! (100%)")
        overall_status.success("Setup complete. All models are ready for analysis.")
        time.sleep(1.0)  
        overall_progress.empty()
        overall_status.empty()

        st.session_state.models_ready = True
        st.session_state["_loaded_models"] = models
        st.session_state["_loaded_parser"] = parser
    else:
        models = st.session_state.get("_loaded_models")
        parser = st.session_state.get("_loaded_parser")
        if models is None or parser is None:
        
            models = load_models_with_progress()
            parser = get_parser()
            st.session_state["_loaded_models"] = models
            st.session_state["_loaded_parser"] = parser

    st.markdown("---")
    st.markdown("## Analysis Tool")

    uploaded_files = st.file_uploader("Upload Clinical Trial PDF", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) > 1:
            st.markdown('<div style="color: #e67e22; font-size: 0.85rem; margin-bottom: 10px;">⚠️ You can process only 1 RCT pdf at a single time. Please upload a single file.</div>', unsafe_allow_html=True)

        if st.button("Analyze Document", type="primary"):
            results = []
            progress = st.progress(0, text="Initializing...")
            status = st.empty()

            file_to_process = uploaded_files[0]
            fname = file_to_process.name

            status.markdown(f"**Processing: {fname}**")
            progress.progress(0.1, text="Reading PDF...")

            print(f"[RCT-Reviewer]  Processing PDF: {fname} - Reading and parsing PDF...")
            log.info(f" Processing PDF: {fname} - Reading and parsing PDF...")


            try:
                pdf_bytes = file_to_process.getvalue()
                parsed_data = parser.parse(pdf_bytes)

                num_sentences = len(parsed_data['sentences'])
                print(f"[RCT-Reviewer]  PDF parsed successfully: {fname} - {num_sentences} sentences extracted")
                log.info(f"PDF parsed successfully: {fname} - {num_sentences} sentences extracted")
          

                progress.progress(0.3, text="PDF parsed. Checking content...")

                if not parsed_data['sentences']:
                    st.error(f"Could not extract text from {fname}")
                    progress.progress(1.0, text="Failed - no text extracted")
                    status.markdown(" Analysis failed: could not extract text.")
                    print(f"[RCT-Reviewer] ❌ Failed to extract text from PDF: {fname}")
                    log.error(f"❌ Failed to extract text from PDF: {fname}")
            
                else:

                    print(f"[RCT-Reviewer]  Running RCT classification model on: {fname}...")
                    log.info(f"Running RCT classification model on: {fname}...")
                 

                    progress.progress(0.4, text="Running RCT classification...")
                    status.markdown(f"**Running ML analysis on {fname}...**")

                    rct_res = models['rct'].predict(parsed_data['title'], parsed_data['abstract'])

                    rct_label = "Yes" if rct_res['is_rct'] else "No"
                    print(f"[RCT-Reviewer] RCT classification complete for: {fname} - Is RCT: {rct_label} (Score: {rct_res['score']:.3f})")
                    log.info(f"RCT classification complete for: {fname} - Is RCT: {rct_label} (Score: {rct_res['score']:.3f})")
        

                    print(f"[RCT-Reviewer] Running PICO extraction model on: {fname}...")
                    log.info(f"Running PICO extraction model on: {fname}...")
         

                    progress.progress(0.6, text="Running PICO extraction...")
                    pico_res = models['pico'].annotate(parsed_data['sentences'])

                    pico_counts = {p['domain']: len(p.get('text', [])) for p in pico_res}
                    print(f"[RCT-Reviewer] PICO extraction complete for: {fname} - Extracted: {pico_counts}")
                    log.info(f"PICO extraction complete for: {fname} - Extracted: {pico_counts}")
           

                    print(f"[RCT-Reviewer] Running Risk of Bias assessment model on: {fname}...")
                    log.info(f" Running Risk of Bias assessment model on: {fname}...")
                

                    progress.progress(0.8, text="Running Risk of Bias assessment...")
                    bias_res = models['bias'].annotate(parsed_data['sentences'], parsed_data['text'])

                    bias_summary = {b['domain']: b.get('judgement', 'N/A') for b in bias_res}
                    print(f"[RCT-Reviewer] Risk of Bias assessment complete for: {fname} - Judgements: {bias_summary}")
                    log.info(f"Risk of Bias assessment complete for: {fname} - Judgements: {bias_summary}")
               

                    progress.progress(1.0)
                    status.markdown(" Analysis complete!")

                    print(f"[RCT-Reviewer] ✅ Full analysis complete for PDF: {fname}")
                    log.info(f"✅ Full analysis complete for PDF: {fname}")
                  

                    result = {
                        "filename": fname, "pdf_bytes": pdf_bytes,
                        "rct": rct_res, "pico": pico_res, "bias": bias_res, "parsed": parsed_data
                    }
                    results.append(result)
            except Exception as e:
                progress.progress(1.0, text="Failed with error")
                status.markdown(f"❌ Analysis failed: {str(e)}")
                st.error(f"Error processing {fname}: {str(e)}")
                print(f"[RCT-Reviewer] ❌ Error processing PDF: {fname} - {str(e)}")
                log.error(f"❌ Error processing PDF: {fname} - {str(e)}")
             

            st.session_state['results'] = results

    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']

        for result in results:

            st.divider()

            st.markdown("###  RCT Classification")
            rct = result['rct']
            rct_col1, rct_col2, rct_col3 = st.columns(3)
            with rct_col1:
                st.metric("Is uploaded file an RCT?", "Yes" if rct['is_rct'] else "No", delta=f"Score: {rct['score']:.3f}")
            with rct_col2:
                st.metric("Probability", f"{rct['probability']:.1%}")
            with rct_col3:
                st.metric("Model", rct.get('model', 'SVM'))

            st.markdown("---")
            st.markdown("###  Risk of Bias Assessment (Analysis Only precise for RCTs)")
            bias = result.get('bias', [])

            if bias:
                bias_data = []
                for b in bias:
                    raw_judgement = b.get('judgement', 'N/A')
                    if raw_judgement == 'low':
                        judgement = 'Low'
                    else:
                        judgement = 'High/Unclear'
                    bias_data.append({
                        "Domain": b['domain'],
                        "Judgement": judgement,
                    })

                df_bias = pd.DataFrame(bias_data)

                def color_judgement(val):
                    if val == 'Low':
                        return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    else:
                        return 'background-color: #f8d7da; color: #721c24; font-weight: bold'

                try:
                    styled_df = df_bias.style.map(color_judgement, subset=['Judgement'])
                except Exception:
                    styled_df = df_bias.style.applymap(color_judgement, subset=['Judgement'])

                st.dataframe(styled_df, width="stretch", hide_index=True)

                st.markdown("#### Risk of Bias Assessment Evidence")
                for b in bias:
                    domain = b.get('domain', '')
                    color = BIAS_COLORS.get(domain, (1.0, 0.3, 0.3))
                    hex_color = '#%02x%02x%02x' % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                    icon = "🟢" if b.get('judgement') == 'low' else "🔴"
                    raw_j = b.get('judgement', 'N/A')
                    display_j = 'Low' if raw_j == 'low' else 'High/Unclear'

                    with st.expander(f"{icon} {domain}"):
                        if raw_j == 'low':
                            st.markdown(f"**Judgement:** <span style='background-color:#28a745;padding:2px 8px;border-radius:3px;color:white;font-weight:bold'>{display_j}</span>", unsafe_allow_html=True)
                        else:
                            fg_color = 'white' if sum(color) < 1.5 else 'black'
                            st.markdown(f"**Judgement:** <span style='background-color:{hex_color};padding:2px 8px;border-radius:3px;color:{fg_color};font-weight:bold'>{display_j}</span>", unsafe_allow_html=True)

                        st.markdown(f"These are the sentences extracted from the uploaded RCT which depict particular **{domain}**.")
                        st.markdown("**Evidence:**")
                        if b.get('text'):
                            for i, evidence in enumerate(b['text'], 1):
                                render_evidence_item(i, evidence)
                        else:
                            st.caption("_No evidence sentences found_")
            else:
                st.caption("_No Risk of Bias assessment could be generated._")

            st.markdown("---")
            st.markdown("###  PICO Extraction")
            pico = result.get('pico', [])
            pico_icons = {"Population": "P - ", "Intervention": "I - ", "Outcomes": "O - "}

            for pico_domain in ["Population", "Intervention", "Outcomes"]:
                with st.expander(f"{pico_icons.get(pico_domain)} {pico_domain}"):
                    st.markdown(f"These are the sentences extracted from the uploaded RCT which depict particular **{pico_domain}**.")
                    domain_data = next((p for p in pico if p['domain'] == pico_domain), None)
                    if domain_data and domain_data.get('text'):
                        for i, sent in enumerate(domain_data['text'], 1):
                            render_evidence_item(i, sent)
                    else:
                        st.caption("_No elements extracted_")

            st.markdown("---")
            st.markdown("#### Download Highlighted PDFs / Results")
            st.caption("📌 Note: For optimal accuracy, use this together with the separately extracted evidence PDFs. Highlighting may not work in some PDFs due to differences in text formatting.")

            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                if st.button(" Generate Highlighted Bias PDF", key=f"bias_pdf_{result['filename']}"):
                    fname = result['filename']
                    print(f"[RCT-Reviewer] Generating Bias highlighted PDF for: {fname}...")
                    log.info(f"Generating Bias highlighted PDF for: {fname}...")

                    with st.spinner("Creating Bias-annotated PDF with highlights & superscripts... Please be patient... Precision takes time..."):
                        annotations = []
                        for b in result.get('bias', []):
                            for text in b.get('text', []):
                                annotations.append({"text": text, "type": "bias", "bias_domain": b.get('domain', '')})
                        bias_pdf = create_bias_highlighted_pdf(result['pdf_bytes'], annotations)
                        print(f"[RCT-Reviewer] ✅ Bias highlighted PDF generated successfully for: {fname}")
                        log.info(f"✅ Bias highlighted PDF generated successfully for: {fname}")
  
                        st.download_button(" Download Highlighted Bias PDF", bias_pdf, f"bias_{result['filename']}", "application/pdf", key=f"dl_bias_{result['filename']}")

            with dl_col2:
                if st.button(" Generate Highlighted PICO PDF", key=f"pico_pdf_{result['filename']}"):
                    fname = result['filename']
                    print(f"[RCT-Reviewer] Generating PICO highlighted PDF for: {fname}...")
                    log.info(f"Generating PICO highlighted PDF for: {fname}...")
            
                    with st.spinner("Creating PICO-annotated PDF with highlights & superscripts... Please be patient... Precision takes time..."):
                        annotations = []
                        for p in result.get('pico', []):
                            for text in p.get('text', []):
                                annotations.append({"text": text, "type": p['domain']})
                        pico_pdf = create_pico_highlighted_pdf(result['pdf_bytes'], annotations)
                        print(f"[RCT-Reviewer] ✅ PICO highlighted PDF generated successfully for: {fname}")
                        log.info(f"✅ PICO highlighted PDF generated successfully for: {fname}")
                    
                        st.download_button(" Download Highlighted PICO PDF", pico_pdf, f"pico_{result['filename']}", "application/pdf", key=f"dl_pico_{result['filename']}")

            dl_col3, dl_col4 = st.columns(2)

            with dl_col3:
                if st.button(" Generate Bias Evidence PDF", key=f"bias_ev_{result['filename']}"):
                    fname = result['filename']
                    print(f"[RCT-Reviewer] Generating Bias Evidence PDF for: {fname}...")
                    log.info(f"Generating Bias Evidence PDF for: {fname}...")
                 
                    bias_ev_pdf = create_bias_evidence_pdf(result.get('bias', []), result['filename'])
                    print(f"[RCT-Reviewer] ✅ Bias Evidence PDF generated successfully for: {fname}")
                    log.info(f"✅ Bias Evidence PDF generated successfully for: {fname}")
             
                    st.download_button(" Download Bias Evidence", bias_ev_pdf, f"bias_evidence_{result['filename']}", "application/pdf", key=f"dl_bias_ev_{result['filename']}")

            with dl_col4:
                if st.button(" Generate PICO Evidence PDF", key=f"pico_ev_{result['filename']}"):
                    fname = result['filename']
                    print(f"[RCT-Reviewer] Generating PICO Evidence PDF for: {fname}...")
                    log.info(f"Generating PICO Evidence PDF for: {fname}...")
                   
                    pico_ev_pdf = create_pico_evidence_pdf(result.get('pico', []), result['filename'])
                    print(f"[RCT-Reviewer] ✅ PICO Evidence PDF generated successfully for: {fname}")
                    log.info(f"✅ PICO Evidence PDF generated successfully for: {fname}")
             
                    st.download_button(" Download PICO Evidence", pico_ev_pdf, f"pico_evidence_{result['filename']}", "application/pdf", key=f"dl_pico_ev_{result['filename']}")

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            if st.button(" Export JSON"):
                print(f"[RCT-Reviewer] Exporting results to JSON...")
                log.info(f"Exporting results to JSON...")

                json_data = export_to_json(results)
                print(f"[RCT-Reviewer] ✅ JSON export complete")
                log.info(f"✅ JSON export complete")
                
                st.download_button("Download JSON", json_data, f"rct_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")

        with exp_col2:
            if st.button(" Export CSV"):
                print(f"[RCT-Reviewer] Exporting results to CSV...")
                log.info(f"Exporting results to CSV...")
   
                csv_data = export_to_csv(results)
                print(f"[RCT-Reviewer] CSV export complete")
                log.info(f"CSV export complete")
  
                st.download_button("Download CSV", csv_data, f"rct_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")



    st.markdown("---")
    st.markdown("## Citation")
    st.markdown('<p style="margin:0; color:#555; font-size:1.1rem;"><i>If you use RCT-Reviewer in your research, please cite both RCT-Reviewer and the original RobotReviewer paper.</i></p>', unsafe_allow_html=True)

    rct_citations = {
        "APA": "Sahu, V. (2026). RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs). Zenodo. https://doi.org/10.5281/zenodo.20618338",
        "Harvard": "Sahu, V., 2026. RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs). Zenodo. Available at: https://doi.org/10.5281/zenodo.20618338",
        "MLA": 'Sahu, Vihaan. "RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs)." 2026, Zenodo, https://doi.org/10.5281/zenodo.20618338.',
        "Chicago": 'Sahu, Vihaan. 2026. "RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs)." Zenodo. https://doi.org/10.5281/zenodo.20618338.',
        "IEEE": 'V. Sahu, "RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs)," Zenodo, 2026. doi: 10.5281/zenodo.20618338.',
        "Vancouver": "Sahu V. RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs). Zenodo. 2026. doi:10.5281/zenodo.20618338"
    }

    robot_citations = {
        "APA": "Marshall, I. J., Kuiper, J., Banner, E., & Wallace, B. C. (2017). Automating Biomedical Evidence Synthesis: RobotReviewer. Proceedings of the Conference of the Association for Computational Linguistics (ACL), 7–12.",
        "Harvard": "Marshall, I.J., Kuiper, J., Banner, E. and Wallace, B.C., 2017. Automating Biomedical Evidence Synthesis: RobotReviewer. Proceedings of the Conference of the Association for Computational Linguistics (ACL), pp.7-12.",
        "MLA": 'Marshall, Iain J., et al. "Automating Biomedical Evidence Synthesis: RobotReviewer." Proceedings of the Conference of the Association for Computational Linguistics (ACL), 2017, pp. 7–12.',
        "Chicago": 'Marshall, Iain J., Joël Kuiper, Edward Banner, and Byron C. Wallace. 2017. "Automating Biomedical Evidence Synthesis: RobotReviewer." Proceedings of the Conference of the Association for Computational Linguistics (ACL), 7–12.',
        "IEEE": 'I. J. Marshall, J. Kuiper, E. Banner, and B. C. Wallace, "Automating Biomedical Evidence Synthesis: RobotReviewer," in Proceedings of the Conference of the Association for Computational Linguistics (ACL), 2017, pp. 7-12.',
        "Vancouver": "Marshall IJ, Kuiper J, Banner E, Wallace BC. Automating Biomedical Evidence Synthesis: RobotReviewer. Proceedings of the Conference of the Association for Computational Linguistics (ACL). 2017:7-12."
    }

    citation_style = st.selectbox(
        "Select citation style",
        ["APA", "Harvard", "MLA", "Chicago", "IEEE", "Vancouver"]
    )

    rct_cite_text = rct_citations[citation_style]
    robot_cite_text = robot_citations[citation_style]


    st.markdown(f'<div class="citation-box"><p style="margin:0;">{rct_cite_text}</p></div>', unsafe_allow_html=True)

    rct_ris = """TY  - JOUR
AU  - Sahu, V
TI  - RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs)
PY  - 2026
DO  - 10.5281/zenodo.20618338
ER  -"""

    rct_bib = """@software{RCT-Reviewer,
  author    = {Sahu, V.},
  title     = {RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20618338},
  url       = {https://doi.org/10.5281/zenodo.20618338}
}"""

    rct_ris_encoded = base64.b64encode(rct_ris.encode()).decode()
    rct_bib_encoded = base64.b64encode(rct_bib.encode()).decode()

    escaped_rct_citation = rct_cite_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    components.html(f"""
    <style>
        .cit-btn {{
            background-color: #5370d6;
            color: white;
            font-weight: 400;
            padding: 0.45rem 0.9rem;
            font-size: 0.8rem;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }}
        .cit-btn:hover {{
            background-color: #4157a5;
            transform: translateY(-2px);
        }}
        .cit-btn-row {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
            margin-bottom: 10px;
        }}
    </style>
    <div class="cit-btn-row">
        <button id="copyButtonRct" class="cit-btn">Copy Citation</button>
        <a download="RCT-Reviewer_citation.ris" href="data:application/x-research-info-systems;base64,{rct_ris_encoded}" class="cit-btn">RIS Format</a>
        <a download="RCT-Reviewer_citation.bib" href="data:application/x-bibtex;base64,{rct_bib_encoded}" class="cit-btn">BibTeX Format</a>
    </div>
    <script>
        document.getElementById("copyButtonRct").addEventListener("click", function() {{
            navigator.clipboard.writeText("{escaped_rct_citation}").then(function() {{
                const button = document.getElementById("copyButtonRct");
                const originalText = button.innerText;
                button.innerText = "Copied!";
                setTimeout(function() {{
                    button.innerText = originalText;
                }}, 2000);
            }}, function(err) {{
                console.error('Could not copy text: ', err);
            }});
        }});
    </script>
    """, height=50)



    st.markdown(f'<div class="citation-box"><p style="margin:0;">{robot_cite_text}</p></div>', unsafe_allow_html=True)

    robot_ris = """TY  - JOUR
AU  - Marshall, IJ
AU  - Kuiper, J
AU  - Banner, E
AU  - Wallace, BC
TI  - Automating Biomedical Evidence Synthesis: RobotReviewer
JO  - Proceedings of the Conference of the Association for Computational Linguistics (ACL)
PY  - 2017
SP  - 7
EP  - 12
ER  -"""

    robot_bib = """@article{RobotReviewer2017,
  title    = {{Automating Biomedical Evidence Synthesis: {{RobotReviewer}}}},
  author   = {Marshall, Iain J and Kuiper, Jo{\"e}l and Banner, Edward and Wallace, Byron C},
  journal  = {Proceedings of the Conference of the Association for Computational Linguistics (ACL)},
  volume   = {2017},
  pages    = {7--12},
  month    = {jul},
  year     = {2017},
}"""

    robot_ris_encoded = base64.b64encode(robot_ris.encode()).decode()
    robot_bib_encoded = base64.b64encode(robot_bib.encode()).decode()

    escaped_robot_citation = robot_cite_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    components.html(f"""
    <style>
        .cit-btn {{
            background-color: #5370d6;
            color: white;
            font-weight: 400;
            padding: 0.45rem 0.9rem;
            font-size: 0.8rem;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }}
        .cit-btn:hover {{
            background-color: #4157a5;
            transform: translateY(-2px);
        }}
        .cit-btn-row {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
            margin-bottom: 10px;
        }}
    </style>
    <div class="cit-btn-row">
        <button id="copyButtonRobot" class="cit-btn">Copy Citation</button>
        <a download="RobotReviewer_citation.ris" href="data:application/x-research-info-systems;base64,{robot_ris_encoded}" class="cit-btn">RIS Format</a>
        <a download="RobotReviewer_citation.bib" href="data:application/x-bibtex;base64,{robot_bib_encoded}" class="cit-btn">BibTeX Format</a>
    </div>
    <script>
        document.getElementById("copyButtonRobot").addEventListener("click", function() {{
            navigator.clipboard.writeText("{escaped_robot_citation}").then(function() {{
                const button = document.getElementById("copyButtonRobot");
                const originalText = button.innerText;
                button.innerText = "Copied!";
                setTimeout(function() {{
                    button.innerText = originalText;
                }}, 2000);
            }}, function(err) {{
                console.error('Could not copy text: ', err);
            }});
        }});
    </script>
    """, height=50)

    st.markdown("---")
    st.markdown("##  Acknowledgements")
    st.markdown("""
    RCT-Reviewer is a modernized version of the original [RobotReviewer](https://github.com/ijmarshall/robotreviewer). I extend my sincere gratitude to the original authors: **Iain J. Marshall, Joël Kuiper, Edward Banner, and Byron C. Wallace** for their foundational work in biomedical NLP and for releasing the project as open-source.

    I would also like to thank all contributors and collaborators involved in the RobotReviewer ecosystem, including the Cochrane Crowd and the research teams at UPenn, Northeastern, and UCL, whose efforts in data collection and model development made this tool possible.

    Additionally, I would like to acknowledge the use of [RikaiCode](https://rikaicode.github.io) (Code Repository Context Generator) and [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7), which were invaluable in analyzing and understanding the complex logic of the original [RobotReviewer](https://github.com/ijmarshall/robotreviewer) codebase, as well as assisting in the development and modernization of RobotReviewer.
    """)


    st.markdown("---")

    st.markdown("### References")
    st.markdown("""
    <a id="ref-1"></a>1. Marshall IJ, Kuiper J, Wallace BC. RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials. Journal of the American Medical Informatics Association. 2016;23(1):193-201. [doi](http://dx.doi.org/10.1093/jamia/ocv044)

    <a id="ref-2"></a>2. Soboczenski F, et al. Machine learning to help researchers evaluate biases in clinical trials: a prospective, randomized user study. BMC Medical Informatics and Decision Making. 2019;19(1):96. [doi](http://dx.doi.org/10.1186/s12911-019-0814-z)

    <a id="ref-3"></a>3. Nussbaumer-Streit B, et al. Automating risk of bias assessment in systematic reviews: a real-time mixed methods comparison of human researchers to a machine learning system. BMC Medical Research Methodology. 2022;22:160. [doi](http://dx.doi.org/10.1186/s12874-022-01649-y)

    <a id="ref-4"></a>4. Marshall I, Kuiper J, Wallace B. Automating Risk of Bias Assessment for Clinical Trials. IEEE Journal of Biomedical and Health Informatics. 2015;19(4):1406-1412. [doi](http://dx.doi.org/10.1109/JBHI.2015.2431314)
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Related")
    st.markdown("""
    - RCT-Reviewer: https://github.com/aurumz-rgb/RCT-Reviewer
    - RCT-Reviewer Hugging Face: https://huggingface.co/Aurumz/RCT-Reviewer
    - Original RobotReviewer: https://github.com/ijmarshall/robotreviewer
    - RobotReviewer Zenodo: https://zenodo.org/records/6855718
    """)

    st.markdown("""
                This project is a derivative work of [RobotReviewer](https://github.com/ijmarshall/robotreviewer) and is distributed under the *GNU GENERAL PUBLIC LICENSE v3.0.*

    [![GNU GPL v3 License](https://www.gnu.org/graphics/gplv3-with-text-136x68.png)](https://www.gnu.org/licenses/gpl-3.0.en.html)


    """)


    st.markdown(
        f"""
        <div class="fixed-footer">
            <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto;">
                <div class="footer-text">
                    © Vihaan Sahu 2026  –  Redistributed under GNU GPL v3.0
                </div>
                <div>
                    <a href="https://github.com/aurumz-rgb/RCT-Reviewer" target="_blank">
                        GitHub Repository
                    </a>
                </div>
                <div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()







# The reason I made this monolithic code file is that I wanted none of the .py files to share similar .py components (functions) since that would make maintaining RCT-Reviewer long term, quite difficult for me AND I plan to keep updating app2.py in future and not the others.