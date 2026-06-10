"""
PDF upload component for RCT-Reviewer v1.0.0

Provides a drag-and-drop file uploader with validation,
file size limits, duplicate detection, and a visual
summary of selected files before analysis.

Copyright (C) 2026 Vihaan Sahu
Based on RobotReviewer by Iain Marshall, Joel Kuiper, Byron Wallace
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import streamlit as st



@dataclass(frozen=True)
class UploadedFile:
    """Validated uploaded PDF ready for processing."""
    bytes: bytes
    filename: str
    sha256: str
    size_mb: float



def _compute_sha256(data: bytes) -> str:
    """Return hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _is_valid_pdf(data: bytes) -> bool:
    """Check that *data* starts with the PDF magic bytes."""
    return data[:5] == b"%PDF-"



def pdf_uploader(
    key: str = "pdf_uploader",
    max_files: int = 20,
    max_size_mb: int = 50,
) -> list[tuple[bytes, str]] | None:
    """Render the PDF upload UI and return validated files.

    Parameters
    ----------
    key : str
        Unique Streamlit widget key (change if you embed the uploader
        more than once on the same page).
    max_files : int
        Maximum number of PDF files the user can upload at once.
    max_size_mb : int
        Per-file size limit in megabytes.

    Returns
    -------
    list[tuple[bytes, str]] | None
        A list of ``(pdf_bytes, filename)`` tuples, or ``None`` when
        nothing has been uploaded yet.

    Notes
    -----
    Files are deduplicated by SHA-256 hash so that the same PDF
    uploaded twice is only processed once.
    """

  
    st.markdown("### 📄 Upload Clinical Trial PDFs")
    st.markdown(
        "Upload PDF files of clinical trial reports for automatic analysis.  "
        "RCT-Reviewer will extract **PICO** data and assess **Risk of Bias** "
        "using transformer models."
    )

    raw_files = st.file_uploader(
        label="Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=key,
        label_visibility="collapsed",
    )

   
    if not raw_files:
        st.info("👈 Click to browse or drag & drop PDF files here")
        return None

   
    validated: list[UploadedFile] = []
    warnings: list[str] = []
    seen_hashes: set[str] = set()

    for f in raw_files:
  
        data = f.read()
        f.seek(0)  

  
        if not f.name.lower().endswith(".pdf"):
            warnings.append(f"⚠️ Skipped `{f.name}` — not a PDF file")
            continue

   
        if not _is_valid_pdf(data):
            warnings.append(
                f"⚠️ Skipped `{f.name}` — file does not appear to be a valid PDF"
            )
            continue

     
        size_mb = len(data) / (1024 * 1024)
        if size_mb > max_size_mb:
            warnings.append(
                f"⚠️ Skipped `{f.name}` — too large "
                f"({size_mb:.1f} MB > {max_size_mb} MB limit)"
            )
            continue

   
        if len(validated) >= max_files:
            warnings.append(f"⚠️ Reached maximum of {max_files} files — stopped here")
            break


        file_hash = _compute_sha256(data)
        if file_hash in seen_hashes:
            warnings.append(f"⚠️ Skipped duplicate `{f.name}`")
            continue

        seen_hashes.add(file_hash)
        validated.append(
            UploadedFile(
                bytes=data,
                filename=f.name,
                sha256=file_hash,
                size_mb=round(size_mb, 2),
            )
        )

   
    for w in warnings:
        st.warning(w)

    
    if not validated:
        st.error("No valid PDF files could be accepted. Check the warnings above.")
        return None

  
    with st.expander(f"📁 {len(validated)} file(s) selected", expanded=True):
        _render_file_grid(validated)

    return [(vf.bytes, vf.filename) for vf in validated]






def _render_file_grid(files: list[UploadedFile], columns: int = 4) -> None:
    """Display selected files in a responsive grid of cards."""
    cols = st.columns(min(len(files), columns))

    for i, vf in enumerate(files):
        with cols[i % len(cols)]:

            st.markdown(f"**📄 {vf.filename}**")

            size_color = "normal" if vf.size_mb < 10 else ("orange" if vf.size_mb < 30 else "red")
            st.caption(f"Size: {vf.size_mb} MB")
  
            st.caption(f"Hash: `{vf.sha256[:12]}…`")


def render_file_stats(files: list[UploadedFile]) -> None:
    """Optional: show aggregate statistics about uploaded files."""
    if not files:
        return

    total_size = sum(vf.size_mb for vf in files)
    avg_size = total_size / len(files)

    col1, col2, col3 = st.columns(3)
    col1.metric("Files", len(files))
    col2.metric("Total size", f"{total_size:.1f} MB")
    col3.metric("Avg. size", f"{avg_size:.1f} MB")