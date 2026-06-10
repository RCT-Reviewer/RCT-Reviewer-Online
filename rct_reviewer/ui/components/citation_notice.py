

import streamlit as st


def show_citation_notice() -> None:
    """Display citation and attribution notice in an expandable section."""
    with st.expander("How to Cite & Attribution", expanded=False):
        st.markdown("---")
        st.markdown("### Citing RCT-Reviewer v1.0.0")
        st.markdown("If you use **RCT-Reviewer** (this modified version) in your work, please cite it as follows:")
        st.code(
            "Sahu V. RCT-Reviewer: Modernized automatic extraction of data from "
            "clinical trial reports, version 1.0.0. 2026. Based on RobotReviewer "
            "by Marshall IJ, Kuiper J, Banner E, Wallace BC.",
            language=None,
        )
        st.markdown("**BibTeX:**")
        st.code(
            '@software{rct_reviewer_2026,\n'
            '  author    = {Sahu, Vihaan},\n'
            '  title     = {{RCT-Reviewer}: Modernized Automatic Extraction of Data\n'
            '               from Clinical Trial Reports},\n'
            '  year      = {2026},\n'
            '  version   = {1.0.0},\n'
            '  license   = {GPL-3.0},\n'
            '  note      = {Based on RobotReviewer by Marshall, Kuiper, Wallace (2017)}\n'
            '}',
            language="bibtex",
        )
        st.markdown("---")
        st.markdown("### Original RobotReviewer Citation")
        st.markdown("This software is derived from **RobotReviewer**. Please also cite the original publication:")
        st.code(
            'Marshall IJ, Kuiper J, Banner E, Wallace BC. "Automating Biomedical '
            'Evidence Synthesis: RobotReviewer." Proceedings of the Conference '
            "of the Association for Computational Linguistics (ACL). 2017:7-12.",
            language=None,
        )
        st.markdown("**BibTeX:**")
        st.code(
            '@inproceedings{RobotReviewer2017,\n'
            '  title     = {Automating Biomedical Evidence Synthesis: {RobotReviewer}},\n'
            '  author    = {Marshall, Iain J and Kuiper, Joel and Banner, Edward\n'
            '               and Wallace, Byron C},\n'
            '  booktitle = {Proceedings of ACL},\n'
            '  pages     = {7--12},\n'
            '  year      = {2017}\n'
            '}',
            language="bibtex",
        )
        st.markdown("---")
        st.markdown("### Original RobotReviewer Authors")
        st.markdown(
            "| Name | Email |\n"
            "|------|-------|\n"
            "| Iain Marshall | mail@ijmarshall.com |\n"
            "| Joel Kuiper | me@joelkuiper.com |\n"
            "| Byron Wallace | byron.wallace@utexas.edu |"
        )
        st.markdown(
            "**Original repository:** "
            "[https://github.com/ijmarshall/robotreviewer]"
            "(https://github.com/ijmarshall/robotreviewer)"
        )
        st.markdown("---")
        st.markdown("### Modifications in RCT-Reviewer v1.0.0")
        st.markdown(
            "- Complete modernization to **Python 3.13**\n"
            "- Replacement of **TensorFlow 1.x / Keras 1.x** with **PyTorch / HuggingFace Transformers**\n"
            "- Replacement of **Flask + React** web interface with **Streamlit**\n"
            "- Removal of **Celery + RabbitMQ** task queue\n"
            "- Replacement of **bert-as-service** with native **HuggingFace** embeddings\n"
            "- Modernized data structures from custom MultiDict to **Pydantic v2** models\n"
            "- Modernized configuration from JSON + env vars to **Pydantic Settings**\n"
            "- Addition of **PyMuPDF** as fallback PDF parser\n"
            "- Complete rewrite of the web UI using **Streamlit components**"
        )
        st.markdown("---")
        st.markdown("### Contact")
        st.markdown("**Vihaan Sahu**\n\n- vsahu@seu.edu.ge\n- pteroisvolitans12@gmail.com")
        st.markdown("---")
        st.caption(
            "Both the original RobotReviewer and RCT-Reviewer are distributed "
            "under the **GNU General Public License v3.0 (GPL-3.0)**. "
            "Modifications are copyright (C) 2026 Vihaan Sahu."
        )
