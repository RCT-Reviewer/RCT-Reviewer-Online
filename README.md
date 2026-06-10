# RCT-Reviewer 

**For the time being, This repository allows users to run RCT-Reviewer locally by downloading the required model files using Git Large File Storage (LFS).**


**RCT-Reviewer** is a modernized, standalone version of the acclaimed [RobotReviewer](https://github.com/ijmarshall/robotreviewer) project. It is a pure-Python application designed to run locally via Streamlit, automating the extraction of PICO data and the assessment of Risk of Bias from clinical trial PDF reports.

![Python](https://img.shields.io/badge/python-3.13-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/license-GPL%20v3.0-blue)
![ML](https://img.shields.io/badge/ML-SVM%20(Linear%20Classifier)-orange)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)

---

## 🚀 Key Features

*   **RCT Classification**: Determines if a document is a Randomized Controlled Trial using a Linear SVM model.
*   **PICO Extraction**: Extracts sentences regarding **P**opulation, **I**ntervention, and **O**utcomes.
*   **Risk of Bias Assessment**: Automatically assesses risk across 6 Cochrane domains (Random sequence generation, Allocation concealment, Blinding, etc.).
*   **Modernized Stack**: No Java, no Docker, and no external databases. Built purely in Python 3.13 using Streamlit.
*   **PDF Annotation**: Generates downloadable PDFs with highlights for PICO and Risk of Bias evidence.

---

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.13**: This project is optimized for the latest Python version.
2.  **Git**: For cloning the repository.
3.  **Git LFS (Large File Storage)**: Required because the pre-trained ML model weights (`.h5`, `.pickle`, `.npz`) are large files.

---

## 🛠️ Installation Guide

This application runs **entirely locally**. Follow these steps carefully to set up the environment and download the necessary model weights.

### 1. Clone the Repository
```bash
git clone <this-repo-url>
cd RCT-Reviewer
```

### 2. Install Git LFS
The model weights are stored using Git LFS. You must initialize LFS to pull the actual model files (otherwise you will only download small pointer files).

```bash
git lfs install
```

### 3. Pull Model Weights
Download the heavy model files (CNN weights, SVM weights, and lexicons).

```bash
# Tracks the specific file types used by this project
git lfs track "*.h5"
git lfs track "*.pickle"
git lfs track "*.npz"

# Pull the actual data
git lfs pull
```

> **⚠️ Important**: If you skip `git lfs pull`, the application will crash with `FileNotFoundError` or unpickling errors because the model "files" will be empty text pointers.

### 4. Create Virtual Environment
It is highly recommended to use a virtual environment.

```bash
# Create venv
python3.13 -m venv .venv

# Activate it
# Linux / macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### 5. Install Dependencies
Install the required Python libraries.

```bash
pip install -r requirements.txt
```

### 6. Download NLP Model
The PDF parser uses `spaCy` for sentence segmentation.

```bash
python -m spacy download en_core_web_sm
```

---

## 💻 Usage

Once the setup is complete, launch the Streamlit app:

```bash
streamlit run rct_reviewer/app.py
```

This will open a local web interface in your browser.

### How to Use the App:
1.  **Upload**: Drag and drop clinical trial PDFs (supports batch processing).
2.  **Analyze**: Click "Analyze Documents".
3.  **Review**:
    *   View RCT probability scores.
    *   Read extracted PICO sentences.
    *   Inspect the Risk of Bias table (color-coded).
4.  **Export**: Download annotated PDFs (separate files for PICO and Bias highlights) or export data to JSON/CSV.

---

##  Architecture & Models

This project preserves the machine learning core of the original RobotReviewer while modernizing the infrastructure.

### The ML Pipeline
1.  **PDF Parsing**: Uses **PyMuPDF (fitz)** to extract text and **spaCy** to segment sentences. No Java/GROBID dependency required.
2.  **Feature Extraction**: Uses `HashingVectorizer` (scikit-learn) to convert text into high-dimensional sparse matrices.
3.  **Classification:**
    * MiniClassifier: A lightweight Linear SVM wrapper that loads pre-trained `.npz` weights.
    * SVM-Only Pipeline: CNN models have been removed due to TensorFlow compatibility issues on Python 3.11–3.13 and are not required for accurate predictions.

### Model Files Structure
Ensure these files exist in your `data/` directory after running `git lfs pull`:

```text
data/
├── pico/
│   ├── P_model.npz    # Population classifier weights
│   ├── I_model.npz    # Intervention weights
│   └── O_model.npz    # Outcomes weights
├── bias/
│   ├── bias_sent_level.npz   # Sentence-level evidence finder
│   └── bias_doc_level.npz    # Document-level bias classifier
├── rct/
│   ├── rct_svm_weights.npz   # RCT classifier weights
│   └── *.h5                  # (Optional) CNN weights
└── drugbank/
    └── drugbank.pck          # Drug name lexicon
```

---

## 🔄 Differences from Original RobotReviewer

| Feature | Original RobotReviewer (2017) | RCT-Reviewer |
| :--- | :--- | :--- |
| **Interface** | Flask + React | **Streamlit** (Pure Python) |
| **PDF Parsing** | GROBID (Requires Java/Docker) | **PyMuPDF** (Native Python) |
| **Task Queue** | Celery + RabbitMQ | **Synchronous** (Local execution) |
| **Data Models** | MultiDict | **Pydantic** |
| **Deployment** | Docker Compose | **Local Streamlit Run** |
| **ML Core** | SVM / CNN | **Same Weights** (SVM prioritized) |


---

### NOTE:


This project uses a **Linear SVM-only pipeline** instead of the original SVM + CNN ensemble. CNN models exist only as legacy RobotReviewer artifacts (.h5 files), but are NOT used.

CNN models were removed because:
- They depend on TensorFlow/Keras `.h5` files
- They break on Python 3.11–3.13 due to compatibility issues
- They significantly increase deployment complexity

SVM is used exclusively because:
- It already contains the full predictive signal
- Accuracy loss is negligible (~0–2%)
- It is faster, lighter, and fully stable in local environments
- It ensures reproducibility across all systems


---

## 🐛 Troubleshooting

**Q: I get `FileNotFoundError: data/pico/P_model.npz`**
**A:** You likely skipped the Git LFS step. Run `git lfs pull` to download the actual model weights.

**Q: I get `ModuleNotFoundError: No module named 'rct_reviewer'`**
**A:** Ensure you are running the command from the root directory of the project, or set your `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**Q: `Can't find model 'en_core_web_sm'`**
**A:** You forgot to download the spaCy model. Run:
```bash
python -m spacy download en_core_web_sm
```

**Q: The app is running but predictions look wrong.**
**A:** Check that your `.npz` files are > 1MB in size. If they are tiny text files (bytes), Git LFS did not pull them correctly.

---

## License

This project is a derivative work of [RobotReviewer](https://github.com/ijmarshall/robotreviewer) and is distributed under the  GNU GENERAL PUBLIC LICENSE v3.0.

