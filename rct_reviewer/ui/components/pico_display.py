"""
PICO results display component for RCT-Reviewer.

Copyright (C) 2026 Vihaan Sahu
"""

import streamlit as st

from rct_reviewer.core.models import DocumentAnalysis, PICODomain


# Icons and colors for PICO domains
PICO_ICONS = {
    PICODomain.POPULATION: "👥",
    PICODomain.INTERVENTION: "💊",
    PICODomain.OUTCOMES: "📈",
}

PICO_COLORS = {
    PICODomain.POPULATION: "#2196F3",  
    PICODomain.INTERVENTION: "#4CAF50",  
    PICODomain.OUTCOMES: "#FF9800",  
}


def display_pico_results(documents: list[DocumentAnalysis]) -> None:
    """Display PICO extraction results for multiple documents.
    
    Args:
        documents: List of analyzed documents
    """
    if not documents:
        st.info("No documents to display.")
        return
    
   
    docs_with_pico = [d for d in documents if (d.pico or d.pico_spans) and not d.parse_error]
    
    if not docs_with_pico:
        st.warning("No PICO extractions available.")
        return
    
    st.markdown("### 🎯 PICO Extraction Results")
    st.caption("Automated extraction — should be verified by human reviewers")
    
   
    selected_idx = st.selectbox(
        "📄 Select document:",
        options=range(len(docs_with_pico)),
        format_func=lambda i: docs_with_pico[i].filename,
        key="pico_doc_selector",
    )
    
    doc = docs_with_pico[selected_idx]
    display_single_pico(doc)


def display_single_pico(doc: DocumentAnalysis) -> None:
    """Display PICO results for a single document."""
   
    if doc.pico_spans:
        _display_pico_spans(doc)
    
    if doc.pico:
        _display_pico_sentences(doc)


def _display_pico_spans(doc: DocumentAnalysis) -> None:
    """Display PICO spans (extracted from title/abstract)."""
    st.markdown("#### Extracted PICO Elements")
    
    cols = st.columns(3)
    
    domain_map = {
        "population": (PICODomain.POPULATION, cols[0]),
        "interventions": (PICODomain.INTERVENTION, cols[1]),
        "outcomes": (PICODomain.OUTCOMES, cols[2]),
    }
    
    for key, (domain, col) in domain_map.items():
        spans = doc.pico_spans.get(key, [])
        icon = PICO_ICONS[domain]
        color = PICO_COLORS[domain]
        
        with col:
            st.markdown(f"**{icon} {domain.value}**")
            if spans:
                for span in spans:
                    st.markdown(f"- {span}")
            else:
                st.caption("_No elements extracted_")
    
    # Full PICO table
    _display_pico_table(doc)


def _display_pico_table(doc: DocumentAnalysis) -> None:
    """Display PICO elements in a table format."""
    import pandas as pd
    
    rows = []
    for key, domain in [("population", PICODomain.POPULATION), 
                        ("interventions", PICODomain.INTERVENTION),
                        ("outcomes", PICODomain.OUTCOMES)]:
        spans = doc.pico_spans.get(key, [])
        for i, span in enumerate(spans, 1):
            rows.append({
                "Domain": f"{PICO_ICONS[domain]} {domain.value}",
                "#": i,
                "Extracted Text": span,
            })
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def _display_pico_sentences(doc: DocumentAnalysis) -> None:
    """Display PICO sentences (extracted from full text)."""
    st.markdown("#### Supporting Sentences (Full Text)")
    
    for pico in doc.pico:
        icon = PICO_ICONS.get(pico.domain, "📄")
        
        with st.expander(f"{icon} {pico.domain.value} ({len(pico.sentences)} sentences)"):
            if pico.sentences:
                for i, sent in enumerate(pico.sentences, 1):
                    st.markdown(f"**{i}.** {sent}")
            else:
                st.caption("No sentences extracted for this domain.")
            
            if pico.mesh_terms:
                st.markdown("**Related MeSH Terms:**")
                st.markdown(", ".join(f"`{term}`" for term in pico.mesh_terms))


def display_pico_comparison(documents: list[DocumentAnalysis]) -> None:
    """Display PICO comparison across multiple documents."""
    import pandas as pd
    
    st.markdown("### 📋 PICO Comparison Across Studies")
    
    rows = []
    for doc in documents:
        if doc.parse_error or not doc.pico_spans:
            continue
        
        row = {"Study": doc.filename[:25] + "..." if len(doc.filename) > 25 else doc.filename}
        
        for key in ["population", "interventions", "outcomes"]:
            spans = doc.pico_spans.get(key, [])
            row[key.title()] = "; ".join(spans[:3]) if spans else "N/A"
        
        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df = df.set_index("Study")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No PICO data available for comparison.")