"""
Export buttons component for RCT-Reviewer.

Copyright (C) 2026 Vihaan Sahu
"""

import json
import csv
import io

import streamlit as st
import pandas as pd

from rct_reviewer.core.models import DocumentAnalysis, BiasDomain


def export_results(documents: list[DocumentAnalysis]) -> None:
    """Display export options for analysis results.
    
    Args:
        documents: List of analyzed documents
    """
    if not documents:
        st.info("No results to export.")
        return
    
    st.markdown("### 📥 Export Results")
    
    cols = st.columns(4)
    
    with cols[0]:
        if st.button("📄 JSON", use_container_width=True):
            _download_json(documents)
    
    with cols[1]:
        if st.button("📊 CSV Summary", use_container_width=True):
            _download_csv_summary(documents)
    
    with cols[2]:
        if st.button("📊 CSV Bias", use_container_width=True):
            _download_csv_bias(documents)
    
    with cols[3]:
        if st.button("📊 CSV PICO", use_container_width=True):
            _download_csv_pico(documents)


def _download_json(documents: list[DocumentAnalysis]) -> None:
    """Download results as JSON."""
    data = []
    for doc in documents:
        doc_dict = doc.model_dump(mode="json")
        # Convert UUID to string
        doc_dict["document_id"] = str(doc_dict["document_id"])
        data.append(doc_dict)
    
    json_str = json.dumps(data, indent=2, default=str)
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name="rct_reviewer_results.json",
        mime="application/json",
    )


def _download_csv_summary(documents: list[DocumentAnalysis]) -> None:
    """Download summary CSV."""
    rows = []
    for doc in documents:
        row = {
            "filename": doc.filename,
            "parsed": not doc.parse_error,
            "is_rct": doc.rct.is_rct if doc.rct else None,
            "rct_score": doc.rct.score if doc.rct else None,
            "rct_probability": doc.rct.probability if doc.rct else None,
            "title": doc.publication.title,
            "journal": doc.publication.journal,
            "year": doc.publication.year,
            "authors": "; ".join(f"{a.lastname} {a.initials}" for a in doc.publication.authors) if doc.publication.authors else None,
            "trial_registry_id": doc.publication.trial_registry_id,
            "processing_time_seconds": doc.processing_time_seconds,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_str = df.to_csv(index=False)
    
    st.download_button(
        label="Download CSV",
        data=csv_str,
        file_name="rct_reviewer_summary.csv",
        mime="text/csv",
    )


def _download_csv_bias(documents: list[DocumentAnalysis]) -> None:
    """Download bias assessment CSV."""
    rows = []
    for doc in documents:
        if doc.parse_error or not doc.bias:
            continue
        for bias in doc.bias:
            rows.append({
                "filename": doc.filename,
                "domain": bias.domain.value,
                "judgement": bias.judgement.value,
                "evidence_sentence_1": bias.sentences[0] if len(bias.sentences) > 0 else None,
                "evidence_sentence_2": bias.sentences[1] if len(bias.sentences) > 1 else None,
                "evidence_sentence_3": bias.sentences[2] if len(bias.sentences) > 2 else None,
            })
    
    if not rows:
        st.warning("No bias data to export.")
        return
    
    df = pd.DataFrame(rows)
    csv_str = df.to_csv(index=False)
    
    st.download_button(
        label="Download CSV",
        data=csv_str,
        file_name="rct_reviewer_bias.csv",
        mime="text/csv",
    )


def _download_csv_pico(documents: list[DocumentAnalysis]) -> None:
    """Download PICO extraction CSV."""
    rows = []
    for doc in documents:
        if doc.parse_error or not doc.pico_spans:
            continue
        
        for domain_key in ["population", "interventions", "outcomes"]:
            spans = doc.pico_spans.get(domain_key, [])
            for i, span in enumerate(spans, 1):
                rows.append({
                    "filename": doc.filename,
                    "domain": domain_key,
                    "element_number": i,
                    "extracted_text": span,
                })
    
    if not rows:
        st.warning("No PICO data to export.")
        return
    
    df = pd.DataFrame(rows)
    csv_str = df.to_csv(index=False)
    
    st.download_button(
        label="Download CSV",
        data=csv_str,
        file_name="rct_reviewer_pico.csv",
        mime="text/csv",
    )