"""
Risk of Bias table component for RCT-Reviewer.

Copyright (C) 2026 Vihaan Sahu
"""

import streamlit as st
import pandas as pd

from rct_reviewer.core.models import BiasAnnotation, BiasDomain, BiasJudgement, DocumentAnalysis


# Color mapping for bias judgements
BIAS_COLORS = {
    BiasJudgement.LOW: "🟢",
    BiasJudgement.HIGH_UNCLEAR: "🔴",
}

BIAS_BG_COLORS = {
    BiasJudgement.LOW: "background-color: #d4edda",
    BiasJudgement.HIGH_UNCLEAR: "background-color: #f8d7da",
}


def display_bias_table(documents: list[DocumentAnalysis]) -> None:
    """Display Risk of Bias summary table for multiple documents.
    
    Args:
        documents: List of analyzed documents
    """
    if not documents:
        st.info("No documents to display.")
        return
    
    
    docs_with_bias = [d for d in documents if d.bias and not d.parse_error]
    
    if not docs_with_bias:
        st.warning("No Risk of Bias assessments available.")
        return
    
    st.markdown("### 📊 Risk of Bias Summary (Cochrane RoB 1.0)")
    st.caption("Automated assessment — should be verified by human reviewers")
    

    domains = [d.value for d in BiasDomain]
    domain_short = [
        "Random\nSequence",
        "Allocation\nConcealment",
        "Blinding:\nParticipants",
        "Blinding:\nOutcome",
        "Incomplete\nOutcome Data",
        "Selective\nReporting",
    ]
    
    data = []
    for doc in docs_with_bias:
        row: dict[str, str] = {"Document": doc.filename[:30] + "..." if len(doc.filename) > 30 else doc.filename}
        for bias in doc.bias:
            icon = BIAS_COLORS.get(bias.judgement, "⚪")
            row[bias.domain.value] = f"{icon} {bias.judgement.value}"
        data.append(row)
    
    if not data:
        st.warning("No bias data to display.")
        return
    
    df = pd.DataFrame(data)
    df = df.set_index("Document")
    df.columns = domain_short
    

    def style_judgement(val: str) -> str:
        if "low" in val.lower():
            return "color: #155724; font-weight: bold"
        elif "high" in val.lower() or "unclear" in val.lower():
            return "color: #721c24; font-weight: bold"
        return ""
    
    styled_df = df.style.map(style_judgement)
    st.dataframe(styled_df, use_container_width=True, height=min(50 + len(docs_with_bias) * 40, 400))
    
  
    _display_bias_evidence(docs_with_bias)


def _display_bias_evidence(documents: list[DocumentAnalysis]) -> None:
    """Display expandable evidence sections for each document."""
    selected_doc = st.selectbox(
        "📄 View evidence for document:",
        options=range(len(documents)),
        format_func=lambda i: documents[i].filename,
        key="bias_evidence_selector",
    )
    
    doc = documents[selected_doc]
    
    st.markdown("#### Evidence Sentences")
    
    for bias in doc.bias:
        icon = BIAS_COLORS.get(bias.judgement, "⚪")
        
        with st.expander(f"{icon} {bias.domain.value}: **{bias.judgement.value}**"):
            if bias.sentences:
                for i, sent in enumerate(bias.sentences, 1):
                    st.markdown(f"**{i}.** {sent}")
            else:
                st.caption("No evidence sentences extracted.")
            
            if bias.annotations:
                st.caption("📄 *Highlighted in original document*")


def display_single_bias(doc: DocumentAnalysis) -> None:
    """Display bias assessment for a single document."""
    if not doc.bias:
        st.info("No Risk of Bias assessment available for this document.")
        return
    
    st.markdown("### 📊 Risk of Bias Assessment")
    
    cols = st.columns(3)
    for i, bias in enumerate(doc.bias):
        with cols[i % 3]:
            icon = BIAS_COLORS.get(bias.judgement, "⚪")
            st.metric(
                label=bias.domain.value,
                value=f"{icon} {bias.judgement.value}",
            )
    

    for bias in doc.bias:
        icon = BIAS_COLORS.get(bias.judgement, "⚪")
        with st.expander(f"{icon} {bias.domain.value}"):
            if bias.sentences:
                for sent in bias.sentences:
                    st.markdown(f"> {sent}")
            else:
                st.caption("No evidence sentences extracted.")