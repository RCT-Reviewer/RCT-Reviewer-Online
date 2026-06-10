# RCT-Reviewer-Online

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rct-reviewer.streamlit.app)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-GPL%20v3.0-blue)

**RCT-Reviewer-Online** is the deployment repository for the live [RCT-Reviewer Web Application](https://rct-reviewer.streamlit.app). 

This lightweight repository hosts the Streamlit frontend (`app2.py`) which connects directly to the Hugging Face Hub to fetch model weights, enabling the application to run in a cloud environment without storing large binary files in the repository itself.

---

## 📂 Related Repositories

*   **Main Repository**: The full source code, local run options, and model weight files are available in the main repository.
    *   🔗 [aurumz-rgb/RCT-Reviewer](https://github.com/aurumz-rgb/RCT-Reviewer)
    
*   **Model Weights (Hugging Face)**: The pre-trained SVM and lexicon models are hosted on the Hugging Face Hub.
    *   🔗 [Aurumz/RCT-Reviewer](https://huggingface.co/Aurumz/RCT-Reviewer)

*   **Original Project**: This project is a modernized, standalone version of the acclaimed RobotReviewer.
    *   🔗 [ijmarshall/robotreviewer](https://github.com/ijmarshall/robotreviewer)

---

## 📖 Citation

If you use the online tool or the underlying software in your research, please cite both the updated RCT-Reviewer version and the original RobotReviewer paper.

### RCT-Reviewer (This Version)

Sahu, V. (2026). RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs). Zenodo. https://doi.org/10.5281/zenodo.20618338

```bibtex
@software{RCT-Reviewer,
  author    = {Sahu, V.},
  title     = {RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20618338},
  url       = {https://doi.org/10.5281/zenodo.20618338}
}
```

### Original RobotReviewer

Marshall IJ, Kuiper J, Banner E, Wallace BC. “Automating Biomedical Evidence Synthesis: RobotReviewer.” Proceedings of the Conference of the Association for Computational Linguistics (ACL). 2017 (July): 7–12.

```bibtex
@article{RobotReviewer2017,
  title    = "Automating Biomedical Evidence Synthesis: {RobotReviewer}",
  author   = "Marshall, Iain J and Kuiper, Jo{\"e}l and Banner, Edward and Wallace, Byron C",
  journal  = "Proceedings of the Conference of the Association for Computational Linguistics (ACL)",
  volume   = 2017,
  pages    = "7--12",
  month    = jul,
  year     = 2017,
}
```

---

## License

This project is a derivative work of [RobotReviewer](https://github.com/ijmarshall/robotreviewer) and is distributed under the GNU GENERAL PUBLIC LICENSE v3.0.
