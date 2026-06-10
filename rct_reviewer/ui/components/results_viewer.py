"""
Results viewer component for RCT-Reviewer.

Copyright (C) 2026 Vihaan Sahu
"""

import streamlit as st
import pandas as pd
import time

from rct_reviewer.core.models import DocumentAnalysis


def display_results(documents: list[DocumentAnalysis]) -> None:
    """Display complete analysis results.
    
    Args:
        documents: List of analyzed documents
    """
    if not documents:
        st.info("No analysis results to display. Upload and analyze PDFs first.")
        return
    
    
    _display_summary_metrics(documents)
    

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Summary Table",
        "📊 Risk of Bias",
        "🎯 PICO Results",
        "📄 Document Details",
    ])
    
    with tab1:
        _display_summary_table(documents)
    
    with tab2:
        from rct_reviewer.ui.components.bias_table import display_bias_table
        display_bias_table(documents)
    
    with tab3:
        from rct_reviewer.ui.components.pico_display import display_pico_results
        display_pico_results(documents)
    
    with tab4:
        _display_document_details(documents)


def _display_summary_metrics(documents: list[DocumentAnalysis]) -> None:
    """Display summary metrics at the top."""
    total = len(documents)
    successful = sum(1 for d in documents if not d.parse_error)
    errors = sum(1 for d in documents if d.parse_error)
    rcts = sum(1 for d in documents if d.rct and d.rct.is_rct)
    avg_time = (
        sum(d.processing_time_seconds or 0 for d in documents) / max(successful, 1)
    )
    
    cols = st.columns(5)
    
    with cols[0]:
        st.metric("Total Documents", total)
    
    with cols[1]:
        st.metric("Successfully Parsed", successful, delta=None)
    
    with cols[2]:
        st.metric("Parse Errors", errors, delta=None)
    
    with cols[3]:
        st.metric("Identified as RCTs", rcts)
    
    with cols[4]:
        st.metric("Avg. Processing Time", f"{avg_time:.1f}s")


def _display_summary_table(documents: list[DocumentAnalysis]) -> None:
    """Display summary table of all documents."""
    st.markdown("#### Document Summary")
    
    rows = []
    for doc in documents:
        row = {
            "Filename": doc.filename,
            "Parsed": "✅" if not doc.parse_error else "❌",
            "Is RCT": (
                "✅ Yes" if doc.rct and doc.rct.is_rct
                else "❌ No" if doc.rct
                else "⚠️ N/A"
            ),
            "RCT Score": f"{doc.rct.score:.3f}" if doc.rct else "N/A",
            "RCT Prob": f"{doc.rct.probability:.1%}" if doc.rct and doc.rct.probability else "N/A",
            "Title": (doc.publication.title[:50] + "...") if doc.publication.title and len(doc.publication.title) > 50 else (doc.publication.title or "N/A"),
            "Year": doc.publication.year or "N/A",
            "Time (s)": f"{doc.processing_time_seconds:.1f}" if doc.processing_time_seconds else "N/A",
        }
        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=min(50 + len(rows) * 35, 500))
    else:
        st.warning("No data to display.")


def _display_document_details(documents: list[DocumentAnalysis]) -> None:
    """Display detailed view for individual documents."""
    valid_docs = [d for d in documents if not d.parse_error]
    
    if not valid_docs:
        st.warning("No successfully parsed documents to display.")
        return
    
    selected_idx = st.selectbox(
        "📄 Select document for details:",
        options=range(len(valid_docs)),
        format_func=lambda i: valid_docs[i].filename,
        key="detail_doc_selector",
    )
    
    doc = valid_docs[selected_idx]
    

    st.markdown("#### Publication Information")
    pub_cols = st.columns(2)
    
    with pub_cols[0]:
        if doc.publication.title:
            st.markdown(f"**Title:** {doc.publication.title}")
        if doc.publication.journal:
            st.markdown(f"**Journal:** {doc.publication.journal}")
        if doc.publication.year:
            st.markdown(f"**Year:** {doc.publication.year}")
    
    with pub_cols[1]:
        if doc.publication.authors:
            authors_str = ", ".join(
                f"{a.lastname} {a.initials}" for a in doc.publication.authors[:5]
            )
            if len(doc.publication.authors) > 5:
                authors_str += f" et al. ({len(doc.publication.authors)} authors)"
            st.markdown(f"**Authors:** {authors_str}")
        if doc.publication.trial_registry_id:
            st.markdown(f"**Trial ID:** `{doc.publication.trial_registry_id}`")
        if doc.publication.doi:
            st.markdown(f"**DOI:** {doc.publication.doi}")
    
   
    if doc.rct:
        st.markdown("#### RCT Classification")
        rct_cols = st.columns(4)
        
        with rct_cols[0]:
            st.markdown("**Decision:**")
            st.markdown("✅ **Is RCT**" if doc.rct.is_rct else "❌ **Not RCT**")
        
        with rct_cols[1]:
            st.markdown(f"**Score:** {doc.rct.score:.4f}")
        
        with rct_cols[2]:
            st.markdown(f"**Threshold:** {doc.rct.threshold:.2f} ({doc.rct.threshold_type})")
        
        with rct_cols[3]:
            st.markdown(f"**Model:** `{doc.rct.model_used}`")
        
        if doc.rct.probability is not None:
            st.progress(doc.rct.probability, text=f"RCT Probability: {doc.rct.probability:.1%}")
    

    if doc.warnings:
        st.markdown("#### ⚠️ Warnings")
        for warning in doc.warnings:
            st.warning(warning)
    

    if doc.models_used:
        st.markdown("#### 🤖 Models Used")
        st.code("\n".join(f"- {m}" for m in doc.models_used), language=None)