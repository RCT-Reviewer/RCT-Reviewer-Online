![banner](assets/banner_transparent.png)


**RCT-Reviewer** is a modernized, standalone version of [RobotReviewer](https://github.com/ijmarshall/robotreviewer?utm_source=chatgpt.com), designed as a third-party reference tool for Risk of Bias assessment. It builds upon RobotReviewer’s original machine learning models trained on 12,808 randomized controlled trials (RCTs).


---

## ⚛️ Why use RCT-REviewer?

RCT-Reviewer is designed as a **Third-Party Tiebreaker Reference** for systematic reviews. Standard guidelines require two independent human reviewers; when they disagree, this tool provides an instant, objective, and data-driven third opinion to resolve ties.

*   **Near-Human Accuracy**: The system achieves **71.0% accuracy** for Risk of Bias judgments, performing within **<8% of human expert consensus** (which stands at 78.3%) [1].

*   **Highly Precise Extraction**: In a randomized Cochrane user trial, the models demonstrated **87% Precision** and **90% Recall** for identifying the exact text snippets supporting the bias judgment [2].

*   **Validated Acceptance**: Real-world feasibility studies show that human reviewers accept the tool's judgments at a rate equal to that of their human peers (Risk Ratio 1.02) [3].

*   **Rigorous Methodology**: Developed by Marshall, Kuiper, and Wallace, the models were trained on **12,808 clinical trial PDFs** using "distant supervision" to ensure high-quality classification without prohibitive manual labeling costs [1,4].

---

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

## References

1. Marshall IJ, Kuiper J, Wallace BC. RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials. Journal of the American Medical Informatics Association. 2016;23(1):193-201. [doi](http://dx.doi.org/10.1093/jamia/ocv044)

2. Soboczenski F, et al. Machine learning to help researchers evaluate biases in clinical trials: a prospective, randomized user study. BMC Medical Informatics and Decision Making. 2019;19(1):96. [doi](http://dx.doi.org/10.1186/s12911-019-0814-z)

3. Nussbaumer-Streit B, et al. Automating risk of bias assessment in systematic reviews: a real-time mixed methods comparison of human researchers to a machine learning system. BMC Medical Research Methodology. 2022;22:160. [doi](http://dx.doi.org/10.1186/s12874-022-01649-y)

4. Marshall I, Kuiper J, Wallace B. Automating Risk of Bias Assessment for Clinical Trials. IEEE Journal of Biomedical and Health Informatics. 2015;19(4):1406-1412. [doi](http://dx.doi.org/10.1109/JBHI.2015.2431314)


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
