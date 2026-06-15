# Author:
#   Vihaan Sahu <pteroisvolitans12@gmail.com>

# This .py file downloads models from Hugging Face hub. This is the default recommended mode which is hosted online.

import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import logging
import base64
import streamlit.components.v1 as components
import pandas as pd
import fitz
import io
import numpy as np
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


def download_models():
    check_file = MODELS_DIR / "pico" / "P_model.npz"

    if check_file.exists():
        log.info("Models already exist in cache.")
        return True

    msg = st.empty()
    msg.info("⬇️ Models not found locally. Downloading from Hugging Face Hub (One-time setup)...")

    from huggingface_hub import snapshot_download
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="model",
                local_dir=MODELS_DIR,
                max_workers=1
            )
            msg.success(f"✅ Models downloaded successfully to: {MODELS_DIR}")
            return True

        except ImportError:
            msg.error("❌ `huggingface_hub` library not found. Please add it to requirements.txt.")
            return False
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                msg.warning(f"⚠️ Download attempt {retry_count} failed: {str(e)[:100]}... Retrying in 5s...")
                time.sleep(5)
            else:
                msg.error(f"❌ Failed to download models after {max_retries} attempts: {e}")
                return False


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
def load_models():
    return {
        "rct": RCTRobot(),
        "pico": PICORobot(),
        "bias": BiasRobot()
    }


@st.cache_resource
def get_parser():
    return PDFParser()


PICO_COLORS = {
    "Population": (1.0, 0.84, 0.0),
    "Intervention": (1.0, 0.6, 0.2),
    "Outcomes": (1.0, 0.5, 0.31),
}

BIAS_COLORS = {
    "Random sequence generation": (1.0, 0.6, 0.6),
    "Allocation concealment": (1.0, 0.3, 0.3),
    "Blinding of participants and personnel": (0.86, 0.08, 0.24),
    "Blinding of outcome assessment": (0.8, 0.4, 0.0),
    "Incomplete outcome data": (0.8, 0.2, 0.2),
    "Selective reporting": (0.5, 0.0, 0.0),
}


def create_bias_highlighted_pdf(pdf_bytes, annotations):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    style_map = {
        "Random sequence generation": "highlight",
        "Allocation concealment": "underline",
        "Blinding of participants and personnel": "squiggly",
        "Blinding of outcome assessment": "manual_bg",
        "Incomplete outcome data": "manual_outline",
        "Selective reporting": "manual_thick_line",
    }

    for page_num, page in enumerate(doc):
        rect = page.rect
        header_height = 95
        header_rect = fitz.Rect(0, 0, rect.width, header_height)
        page.draw_rect(header_rect, color=(1, 1, 1), fill=(1, 1, 1))

        page.insert_text(fitz.Point(10, 15), f"Generated by RCT-Reviewer (Cloud) on {current_date}", fontsize=10, color=(0.2, 0.2, 0.2), fontname="helv")
        page.draw_line(fitz.Point(0, 22), fitz.Point(rect.width, 22), color=(0.8, 0.8, 0.8), width=1)
        page.insert_text(fitz.Point(10, 38), "BIAS HIGHLIGHTS (Distinct Styles):", fontsize=8, color=(0.3, 0.3, 0.3), fontname="helv")

        bias_legend = [
            ("1. Random Seq", (1.0, 0.6, 0.6), "highlight"), ("2. Allocation", (1.0, 0.3, 0.3), "underline"),
            ("3. Blind-Part", (0.86, 0.08, 0.24), "squiggly"), ("4. Blind-Outcome", (0.8, 0.4, 0.0), "manual_bg"),
            ("5. Incomplete", (0.8, 0.2, 0.2), "manual_outline"), ("6. Selective Rep", (0.5, 0.0, 0.0), "manual_thick_line"),
        ]

        legend_x = 10
        legend_y = 52
        for label, color, style in bias_legend[:3]:
            if style == "highlight":
                box_rect = fitz.Rect(legend_x, legend_y - 6, legend_x + 15, legend_y + 6)
                page.draw_rect(box_rect, color=color, fill=color)
            elif style == "underline":
                page.draw_line(fitz.Point(legend_x, legend_y + 4), fitz.Point(legend_x + 15, legend_y + 4), color=color, width=2)
            elif style == "squiggly":
                page.draw_line(fitz.Point(legend_x, legend_y + 4), fitz.Point(legend_x + 15, legend_y + 4), color=color, width=1)
                page.draw_line(fitz.Point(legend_x, legend_y + 2), fitz.Point(legend_x + 15, legend_y + 2), color=color, width=1)
            page.insert_text(fitz.Point(legend_x + 20, legend_y + 3), label, fontsize=6, color=(0.2, 0.2, 0.2), fontname="helv")
            legend_x += 130

        legend_x = 10
        legend_y = 70
        for label, color, style in bias_legend[3:]:
            if style == "manual_bg":
                box_rect = fitz.Rect(legend_x, legend_y - 6, legend_x + 15, legend_y + 6)
                page.draw_rect(box_rect, color=color, fill=color)
            elif style == "manual_outline":
                box_rect = fitz.Rect(legend_x, legend_y - 6, legend_x + 15, legend_y + 6)
                page.draw_rect(box_rect, color=color, width=2)
            elif style == "manual_thick_line":
                page.draw_line(fitz.Point(legend_x, legend_y + 4), fitz.Point(legend_x + 15, legend_y + 4), color=color, width=3)
            page.insert_text(fitz.Point(legend_x + 20, legend_y + 3), label, fontsize=6, color=(0.2, 0.2, 0.2), fontname="helv")
            legend_x += 130

        page.draw_line(fitz.Point(0, header_height - 2), fitz.Point(rect.width, header_height - 2), color=(0.6, 0.6, 0.6), width=1.5)

        bias_annotations = [a for a in annotations if a.get("type") == "bias"]

        for ann in bias_annotations:
            text = ann.get("text", "")
            bias_domain = ann.get("bias_domain", "")
            if not text or len(text) < 10 or not bias_domain:
                continue

            color = BIAS_COLORS.get(bias_domain, (1.0, 0.3, 0.3))
            style = style_map.get(bias_domain, "highlight")

            try:
                areas = page.search_for(text)
                for area in areas:
                    if area.y0 < header_height:
                        continue

                    if style == "highlight":
                        highlight = page.add_highlight_annot(area)
                        highlight.set_colors(stroke=color)
                        highlight.update()
                    elif style == "underline":
                        ul = page.add_underline_annot(area)
                        ul.set_colors(stroke=color)
                        ul.update()
                    elif style == "squiggly":
                        sq = page.add_squiggly_annot(area)
                        sq.set_colors(stroke=color)
                        sq.update()
                    elif style == "manual_bg":
                        bg_rect = fitz.Rect(area.x0 - 2, area.y0 - 1, area.x1 + 2, area.y1 + 1)
                        shape = page.new_shape()
                        shape.draw_rect(bg_rect)
                        shape.finish(color=color, fill=color, fill_opacity=0.3, overlay=False)
                        shape.commit()
                    elif style == "manual_outline":
                        out_rect = fitz.Rect(area.x0 - 1, area.y0 - 1, area.x1 + 1, area.y1 + 1)
                        page.draw_rect(out_rect, color=color, width=1.5)
                    elif style == "manual_thick_line":
                        line_y = area.y1 - 1
                        page.draw_line(fitz.Point(area.x0, line_y), fitz.Point(area.x1, line_y), color=color, width=3)
            except Exception as e:
                log.debug(f"Could not annotate text: {e}")

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def create_pico_highlighted_pdf(pdf_bytes, annotations):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    style_map = {"Population": "highlight", "Intervention": "underline", "Outcomes": "squiggly"}

    for page_num, page in enumerate(doc):
        rect = page.rect
        header_height = 65
        header_rect = fitz.Rect(0, 0, rect.width, header_height)
        page.draw_rect(header_rect, color=(1, 1, 1), fill=(1, 1, 1))

        page.insert_text(fitz.Point(10, 15), f"Generated by RCT-Reviewer (Cloud) on {current_date}", fontsize=10, color=(0.2, 0.2, 0.2), fontname="helv")
        page.draw_line(fitz.Point(0, 22), fitz.Point(rect.width, 22), color=(0.8, 0.8, 0.8), width=1)
        page.insert_text(fitz.Point(10, 40), "PICO HIGHLIGHTS (Distinct Styles):", fontsize=8, color=(0.3, 0.3, 0.3), fontname="helv")

        pico_legend = [
            ("Population (Highlight)", (1.0, 0.84, 0.0), "highlight"),
            ("Intervention (Underline)", (1.0, 0.6, 0.2), "underline"),
            ("Outcomes (Squiggly)", (1.0, 0.5, 0.31), "squiggly"),
        ]

        legend_x = 10
        legend_y = 55
        for label, color, style in pico_legend:
            if style == "highlight":
                box_rect = fitz.Rect(legend_x, legend_y - 6, legend_x + 20, legend_y + 6)
                page.draw_rect(box_rect, color=color, fill=color)
            elif style == "underline":
                page.draw_line(fitz.Point(legend_x, legend_y + 4), fitz.Point(legend_x + 20, legend_y + 4), color=color, width=2)
            elif style == "squiggly":
                page.draw_line(fitz.Point(legend_x, legend_y + 4), fitz.Point(legend_x + 20, legend_y + 4), color=color, width=1)
                page.draw_line(fitz.Point(legend_x, legend_y + 2), fitz.Point(legend_x + 20, legend_y + 2), color=color, width=1)
            page.insert_text(fitz.Point(legend_x + 25, legend_y + 3), label, fontsize=7, color=(0.2, 0.2, 0.2), fontname="helv")
            legend_x += 150

        page.draw_line(fitz.Point(0, header_height - 2), fitz.Point(rect.width, header_height - 2), color=(0.6, 0.6, 0.6), width=1.5)

        pico_annotations = [a for a in annotations if a.get("type") in ["Population", "Intervention", "Outcomes"]]

        for ann in pico_annotations:
            text = ann.get("text", "")
            ann_type = ann.get("type", "")
            if not text or len(text) < 10:
                continue

            color = PICO_COLORS.get(ann_type, (1.0, 0.84, 0.0))
            style = style_map.get(ann_type, "highlight")

            try:
                areas = page.search_for(text)
                for area in areas:
                    if area.y0 < header_height:
                        continue
                    if style == "highlight":
                        highlight = page.add_highlight_annot(area)
                        highlight.set_colors(stroke=color)
                        highlight.update()
                    elif style == "underline":
                        ul = page.add_underline_annot(area)
                        ul.set_colors(stroke=color)
                        ul.update()
                    elif style == "squiggly":
                        sq = page.add_squiggly_annot(area)
                        sq.set_colors(stroke=color)
                        sq.update()
            except Exception as e:
                log.debug(f"Could not annotate text: {e}")

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
        st.image("assets/banner.svg", use_container_width=True)

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
        | **Risk of Bias Accuracy** | ~71.0% agreement accuracy vs expert consensus | Same expected predictive accuracy because the same SVM weights are used |
        | **Supporting Text Precision** | ~87% precision for rationale extraction | Same extraction models retained |
        | **Supporting Text Recall** | ~90% recall | Same extraction models retained |
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
        with st.spinner("Checking / downloading models..."):
            success = download_models()
            if success:
                st.session_state.models_ready = True
            else:
                st.error("Model download failed. Please check logs.")
                st.stop()

    with st.spinner("Loading ML models from cache..."):
        models = load_models()
        parser = get_parser()

    st.markdown("---")
    st.markdown("## Analysis Tool")

    with st.expander("⚙️ Settings"):
        show_evidence = st.checkbox("Show Evidence Sentences", value=True)
        top_k_sentences = st.slider("Evidence Sentences per Domain", 1, 5, 3)

    st.info(f"**Running in Cloud Mode:** Models are loaded from Hugging Face Hub.\nCache location: `{MODELS_DIR}`")

    uploaded_files = st.file_uploader("Upload Clinical Trial PDF", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) > 1:
            st.markdown('<div style="color: #e67e22; font-size: 0.85rem; margin-bottom: 10px;">⚠️ You can process only 1 RCT pdf at a single time. Please upload a single file.</div>', unsafe_allow_html=True)

        if st.button("Analyze Document", type="primary"):
            results = []
            progress = st.progress(0)
            status = st.empty()

            file_to_process = uploaded_files[0]
            
            status.markdown(f"**Processing: {file_to_process.name}**")
            try:
                pdf_bytes = file_to_process.getvalue()
                parsed_data = parser.parse(pdf_bytes)

                if not parsed_data['sentences']:
                    st.error(f"Could not extract text from {file_to_process.name}")
                else:
                    with st.spinner("Running ML analysis..."):
                        rct_res = models['rct'].predict(parsed_data['title'], parsed_data['abstract'])
                        pico_res = models['pico'].annotate(parsed_data['sentences'])
                        bias_res = models['bias'].annotate(parsed_data['sentences'], parsed_data['text'])

                    result = {
                        "filename": file_to_process.name, "pdf_bytes": pdf_bytes,
                        "rct": rct_res, "pico": pico_res, "bias": bias_res, "parsed": parsed_data
                    }
                    results.append(result)
            except Exception as e:
                st.error(f"Error processing {file_to_process.name}: {str(e)}")

            progress.progress(1) 

            status.markdown(" Analysis complete!")
            st.session_state['results'] = results

    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']

        st.markdown("---")
        st.markdown("## Analysis Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", len(results))
        with col2:
            st.metric("Identified RCTs", sum(1 for r in results if r['rct']['is_rct']))
        with col3:
            st.metric("Low Risk Assessments", sum(1 for r in results if any(b['judgement'] == 'low' for b in r['bias'])))
        with col4:
            st.metric("Documents with Issues", len(results) - sum(1 for r in results if r['rct']['is_rct']))

       

        for result in results:

            st.divider()

            st.markdown("###  RCT Classification")
            rct = result['rct']
            rct_col1, rct_col2, rct_col3 = st.columns(3)
            with rct_col1:
                st.metric("Is RCT?", "✅ Yes" if rct['is_rct'] else "❌ No", delta=f"Score: {rct['score']:.3f}")
            with rct_col2:
                st.metric("Probability", f"{rct['probability']:.1%}")
            with rct_col3:
                st.metric("Model", rct.get('model', 'SVM'))

            st.markdown("###  Risk of Bias Assessment")
            bias = result.get('bias', [])

            if bias:
                bias_data = []
                for b in bias:
                    judgement = b.get('judgement', 'N/A')
                    domain = b.get('domain', '')
                    color = BIAS_COLORS.get(domain, (1.0, 0.3, 0.3))
                    hex_color = '#%02x%02x%02x' % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                    bias_data.append({
                        "Domain": b['domain'],
                        "Color": hex_color,
                        "Judgement": judgement,
                        "Evidence": b['text'][0][:60] + "..." if b.get('text') else "N/A"
                    })

                df_bias = pd.DataFrame(bias_data)

                def color_judgement(val):
                    if val == 'low':
                        return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    else:
                        return 'background-color: #f8d7da; color: #721c24; font-weight: bold'

                try:
                    styled_df = df_bias.style.map(color_judgement, subset=['Judgement'])
                except Exception:
                    styled_df = df_bias.style.applymap(color_judgement, subset=['Judgement'])

                st.dataframe(styled_df, width="stretch", hide_index=True)

                if show_evidence:
                    st.markdown("#### Evidence Sentences")
                    for b in bias:
                        domain = b.get('domain', '')
                        color = BIAS_COLORS.get(domain, (1.0, 0.3, 0.3))
                        hex_color = '#%02x%02x%02x' % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                        icon = "🟢" if b.get('judgement') == 'low' else "🔴"

                        with st.expander(f"{icon} {domain}"):
                            ev_col1, ev_col2 = st.columns([1, 2])
                            with ev_col1:
                                st.markdown(f"**Judgement:** `{b.get('judgement', 'N/A')}`")
                            with ev_col2:
                                fg_color = 'white' if sum(color) < 1.5 else 'black'
                                st.markdown(f"**Color:** <span style='background-color:{hex_color};padding:2px 8px;border-radius:3px;color:{fg_color}'>{hex_color}</span>", unsafe_allow_html=True)

                            st.markdown("**Evidence:**")
                            if b.get('text'):
                                for i, evidence in enumerate(b['text'][:top_k_sentences], 1):
                                    st.info(f"{i}. {evidence}")
                            else:
                                st.caption("_No evidence sentences found_")
            else:
                st.caption("_No Risk of Bias assessment could be generated._")

            st.markdown("###  PICO Extraction")
            pico = result.get('pico', [])
            pico_icons = {"Population": "P - ", "Intervention": "I - ", "Outcomes": "O - "}

            for pico_domain in ["Population", "Intervention", "Outcomes"]:
                with st.expander(f"{pico_icons.get(pico_domain, '📄')} {pico_domain}"):
                    domain_data = next((p for p in pico if p['domain'] == pico_domain), None)
                    if domain_data and domain_data.get('text'):
                        for i, sent in enumerate(domain_data['text'][:top_k_sentences], 1):
                            st.info(f"{i}. {sent}")
                    else:
                        st.caption("_No elements extracted_")

            st.markdown("---")
            st.markdown("#### Download Highlighted PDFs / Results")
            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                if st.button(" Generate Bias PDF", key=f"bias_pdf_{result['filename']}"):
                    with st.spinner("Creating Bias-highlighted PDF..."):
                        annotations = []
                        for b in result.get('bias', []):
                            for text in b.get('text', []):
                                annotations.append({"text": text, "type": "bias", "bias_domain": b.get('domain', '')})
                        bias_pdf = create_bias_highlighted_pdf(result['pdf_bytes'], annotations)
                        st.download_button(" Download Bias PDF", bias_pdf, f"bias_{result['filename']}", "application/pdf", key=f"dl_bias_{result['filename']}")

            with dl_col2:
                if st.button(" Generate PICO PDF", key=f"pico_pdf_{result['filename']}"):
                    with st.spinner("Creating PICO-highlighted PDF..."):
                        annotations = []
                        for p in result.get('pico', []):
                            for text in p.get('text', []):
                                annotations.append({"text": text, "type": p['domain']})
                        pico_pdf = create_pico_highlighted_pdf(result['pdf_bytes'], annotations)
                        st.download_button(" Download PICO PDF", pico_pdf, f"pico_{result['filename']}", "application/pdf", key=f"dl_pico_{result['filename']}")

  


        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            if st.button(" Export JSON"):
                json_data = export_to_json(results)
                st.download_button("Download JSON", json_data, f"rct_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")

        with exp_col2:
            if st.button(" Export CSV"):
                csv_data = export_to_csv(results)
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
        "IEEE": 'I. J. Marshall, J. Kuiper, E. Banner, and B. C. Wallace, "Automating Biomedical Evidence Synthesis: RobotReviewer," in Proceedings of the Conference of the Association for Computational Linguistics (ACL), 2017, pp. 7–12.',
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

    Additionally, I acknowledge the use of [RikaiCode](https://rikaicode.github.io) (Code Repository Context Generator), which was invaluable for analyzing and understanding the complex logic of the original RobotReviewer codebase during the modernization process.
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
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()