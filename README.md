# When AI Improves the Text but Changes the Voice: Corpus, Experimental, and Attitudinal Evidence for the Identity Compression Framework

[![DOI](https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/zenodo.22159148.svg)](https://doi.org/10.5281/zenodo.22159148)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-success.svg)](https://github.com/Pegi1727/GenAI-Identity-Compression-Framework)

---

## 🎨 Graphical Abstract

<p align="center">
  <img src="https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/ga.png" alt="Graphical Abstract" width="850"/>
</p>

---

## 📌 Abstract & Overview

This repository contains the complete analytical pipeline, experimental datasets, statistical modeling scripts, and high-resolution figures for the empirical study on GenAI-assisted academic writing. 

While Generative AI tools (e.g., ChatGPT / LLM-based writing assistants) significantly enhance grammatical precision and syntactic complexity, they systematically suppress subjective stance markers, voice uniqueness, and personal authorial identity—a phenomenon conceptualized here as the **Identity Compression Framework**.

---

## 📊 Key Empirical Findings

Paired quasi-experimental analysis ($N = 60$ paired pre/post essays) evaluated via advanced computational corpus linguistics and psychometric scaling (AGS) confirms the central trade-off:

| Metric Category | Feature / Indicator | Pre-AI (Mean ± SD) | Post-AI (Mean ± SD) | Statistical Test ($t$-value) | Effect Size (Cohen's $d$) | $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Linguistic Accuracy** | Grammar Errors ($/100\text{w}$) | $8.42 \pm 2.11$ | $1.15 \pm 0.62$ | $t = -23.04$ | $d = -2.97$ | $< .001$ |
| **Lexical Sophistication** | Lexical Diversity (MTLD) | $48.30 \pm 6.84$ | $82.65 \pm 8.12$ | $t = 29.13$ | $d = 3.76$ | $< .001$ |
| **Syntactic Complexity** | Mean Length of T-unit (MLT) | $12.45 \pm 1.82$ | $18.90 \pm 2.41$ | $t = 16.85$ | $d = 2.18$ | $< .001$ |
| **Authorial Voice** | Stance & Hedging Markers | $6.85 \pm 1.45$ | $2.10 \pm 0.88$ | $t = -19.42$ | $d = -2.51$ | $< .001$ |
| **Psychometric (AGS)** | Perceived Authenticity Gap | $2.18 \pm 0.64$ | $4.42 \pm 0.51$ | $t = 21.80$ | $d = 2.81$ | $< .001$ |

---

## 🖼️ Figures & Framework Visualizations

### 1. Conceptual Framework & Research Design
| Figure | Description |
| :---: | :--- |
| <img src="https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/figure6_identity_compression_framework_pink_gray.png" width="450"/> | **Figure 6** — The Identity Compression Framework (Theoretical Core) |
| <img src="https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/overall_research_design_pink_gray.png" width="450"/> | **Overall Design** — Paired Quasi-Experimental Mixed-Methods Architecture |
| <img src="https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/figure2_data_collection_procedure_pink_gray.png" width="450"/> | **Figure 2** — End-to-End Data Collection Procedure (Raw $\to$ GenAI $\to$ AGS Survey) |
| <img src="https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/figure3_research_workflow_pink_gray.png" width="450"/> | **Figure 3** — Analytical Pipeline Workflow (Modules `01` to `09`) |

### 2. Empirical Findings & Path Modeling
| Figure | Description |
| :---: | :--- |
| <img src="https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/figure4_before_vs_after_ai_pink_gray.png" width="450"/> | **Figure 4** — Contrastive Linguistic Shift Across All Dimensions (Pre vs. Post AI) |
| <img src="https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/figure5_revision_proficiency_interaction_pink_gray.png" width="450"/> | **Figure 5** — Revision Intensity $\times$ L2 Proficiency Interaction Profile |
| <img src="https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/figure7_authenticity_vs_grammar_scatter_pink_gray.png" width="450"/> | **Figure 7** — Grammar Improvement vs. Authorial Authenticity Trade-off |
| <img src="https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/figure8_authenticity_gap_path_diagram_pink_gray.png" width="450"/> | **Figure 8** — Structural Equation / Path Model of the Authenticity Gap |
| <img src="https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/figure9_thematic_map_reduced_agency_pink_gray.png" width="450"/> | **Figure 9** — Qualitative Thematic Map: Agency Dilution & Voice Homogenization |
| <img src="https://raw.githubusercontent.com/Pegi1727/GenAI-Identity-Compression-Framework/main/figures/figure10_integrated_conceptual_model_pink_gray.png" width="450"/> | **Figure 10** — Integrated Computational & Pedagogical Intervention Model |

---
Reproduction & Setup
Prerequisites
Clone the repository and install all required scientific and NLP libraries:

bash
git clone https://github.com/Pegi1727/GenAI-Identity-Compression-Framework.git
cd GenAI-Identity-Compression-Framework
pip install -r requirements.txt
python -m spacy download en_core_web_sm
Running the Pipeline
You can run all analyses sequentially via the command line:

bash
# Execute advanced corpus linguistic and statistical modules
python scripts/06_revision_intensity_and_edits.py
python scripts/07_semantic_affective_shift.py
python scripts/08_corpus_keyness_and_contrastive_viz.py
python scripts/09_syntactic_complexity_lu2010.py
Or open and run Full_Analysis_Pipeline.ipynb in Jupyter Notebook / VS Code for step-by-step verification.

📜 Citation
If you use this dataset, methodology, or the Identity Compression Framework in your research, please cite:

bibtex
@article{merrikhi2026identitycompression,
  title     = {When AI Improves the Text but Changes the Voice: Corpus, Experimental, and Attitudinal Evidence for the Identity Compression Framework},
  author    = {Merrikhi, Pegah},
  journal   = {Journal of Second Language Writing},
  year      = {2026},
  doi       = {10.5281/zenodo.22159148},
  url       = {https://doi.org/10.5281/zenodo.22159148}
}
⚖️ License
This repository is distributed under the MIT License. See the LICENSE file for details.

---
## 📁 Repository Structure
```text
├── data/
│   ├── clean_analysis_df.csv            # Cleaned analytical dataset (60 pairs × 20 features)
│   └── brave world Research_Data.xlsx   # Full raw multi-sheet dataset
├── figures/
│   ├── ga.png                           # Graphical Abstract
│   ├── zenodo.22159148.svg              # Persistent Identifier DOI badge
│   ├── figure2_data_collection_procedure_pink_gray.png
│   ├── figure3_research_workflow_pink_gray.png
│   ├── figure4_before_vs_after_ai_pink_gray.png
│   ├── figure5_revision_proficiency_interaction_pink_gray.png
│   ├── figure6_identity_compression_framework_pink_gray.png
│   ├── figure7_authenticity_vs_grammar_scatter_pink_gray.png
│   ├── figure8_authenticity_gap_path_diagram_pink_gray.png
│   ├── figure9_thematic_map_reduced_agency_pink_gray.png
│   ├── figure10_integrated_conceptual_model_pink_gray.png
│   └── overall_research_design_pink_gray.png
├── scripts/
│   ├── 01_data_cleaning_and_descriptives.py
│   ├── 02_paired_inferential_tests.py
│   ├── 03_lmem_hierarchical_modelling.py
│   ├── 04_path_analysis_structural_model.py
│   ├── 05_qualitative_thematic_analysis.py
│   ├── 06_revision_intensity_and_edits.py      # Word-level Levenshtein (WER) & edit operations
│   ├── 07_semantic_affective_shift.py          # VADER sentiment & Subjectivity profiling
│   ├── 08_corpus_keyness_and_contrastive_viz.py # Log-Likelihood (G²) suppressed/injected keyness
│   └── 09_syntactic_complexity_lu2010.py       # Lu (2010/2011) L2 Syntactic indices (MLS, MLT, C/T)
├── Full_Analysis_Pipeline.ipynb                # Master executable reproduction notebook
├── LICENSE
└── README.md
