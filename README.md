# Evaluating Post-Retrieval Context Noise Filtering in Domain-Specific RAG Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducibility: Saved Data](https://img.shields.io/badge/Reproducibility-Saved--Data%20Checked-brightgreen.svg)]()
[![Target Journal: IEEE Access](https://img.shields.io/badge/Target%20Journal-IEEE%20Access-orange.svg)](https://ieeeaccess.ieee.org)

This repository contains the complete experimental framework, raw execution traces, evaluation records, statistical analysis, and manuscript dossier for the research article:  
**"Evaluating Post-Retrieval Context Noise Filtering in Domain-Specific Retrieval-Augmented Generation Systems"**.

- **Authors**: Baknur Aitimbet, Assylkhan Toilybekov, Ayaulym Zhaskairat  
- **Course**: *Research Methods and Tools* (SE-2418, Astana IT University, 2026)  
- **Target Journal**: *IEEE Access* (Scopus indexed; 2025 CiteScore 9.3, checked 17 November 2026)  

---

## ⚡ 1-Click Reproducibility (Master Pipeline)

To recompute the saved-data tables, non-parametric statistical tests, and publication figures:

```bash
python reproduce_all.py
```

The script always recomputes outputs from `scores.csv` and the de-duplicated
`runs/test_unique.jsonl`. It does **not** rerun retrieval, answer generation,
or LLM judging.

This master pipeline outputs:
- **`report_artifacts/main_metrics.csv`**: Main quality metrics (Hallucination Rate, Faithfulness, Correctness, Precision, Recall) with 95% bootstrap confidence intervals.
- **`report_artifacts/efficiency_metrics.csv`**: Latency, prompt tokens, and chunks retained.
- **`report_artifacts/figure_1_main_metrics.png`**: High-resolution comparison bar chart.
- **`report_artifacts/figure_2_latency_distribution.png`**: Generation latency boxplots.
- **`analysis_output/statistical_comparisons.csv`**: Exact McNemar tests, Wilcoxon signed-rank tests, and Holm-Bonferroni corrections.
- **`analysis_output/hypothesis_decisions.csv`**: Machine-readable hypothesis verification.

---

## 🔬 Experimental Setup

The benchmark evaluates $N = 135$ paired questions from the **RAGBench TechQA** test split (`rungalileo/ragbench`), comprising 405 paired conditions:
1. **Baseline**: Unfiltered Top-10 dense retrieval (`BAAI/bge-small-en-v1.5`).
2. **Length-Matched Control**: Frozen Top-4 dense retrieval (isolates prompt brevity from semantic selection).
3. **Filtered (Proposed Method)**: Cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with relevance threshold $\tau = -2.0$ and cosine deduplication ($> 0.95$).

- **Generator LLM**: `llama3.1:8b` via Ollama (`temperature=0`, `seed=42`, `num_ctx=8192`).
- **Independent Judge LLM**: `qwen2.5:7b` via RAGAS metric specifications in deterministic JSON mode.

---

## 📊 Summary of Main Findings

| Condition | Hallucination Rate (HR) ↓ | Answer Correctness ↑ | Faithfulness ↑ | Prompt Tokens ↓ | Mean Latency (s) ↓ | Retained Chunks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Top-10)** | 82.96% | **0.5663** | 0.5830 | 3,576.06 | 10.73 s | 10.00 |
| **Length-Matched (Top-4)** | **76.30%** | 0.5274 | **0.5844** | **1,514.07** | **5.68 s** | **4.00** |
| **Filtered (Proposed)** | 77.78% | 0.5126 | 0.5289 | 2,090.85 | 7.13 s | 5.59 |

- **Quality**: the observed 5.19 percentage-point hallucination-rate decrease was not significant after Holm correction (adjusted p = 0.474). H1-H3 were not supported; H4 is exploratory.
- **Operational Efficiency**: filtering reduced mean prompt tokens by **41.5%** and mean local generation latency by **33.5%** on 135 unique paired records (paired Wilcoxon p = 1.17e-11).
- **Qualitative Error File**: `error_analysis_completed.csv` contains 35 condition-level rows from 14 questions. All rows have the same saved category, and the file does not contain independent categorical decisions required to verify Cohen's kappa.

---

## 📁 Repository Structure

```text
├── Final_Aitimbet_Toilybekov_Zhaskairat.docx # Complete Master Manuscript & Submission Dossier
├── reproduce_all.py            # Master 1-click reproduction pipeline
├── rag_filter_pipeline.py      # Core RAG retrieval, filtering, and generation pipeline
├── fast_evaluate.py            # Deterministic RAGAS metric evaluator using local Qwen-2.5 judge
├── consolidated_analysis.py    # Statistical hypothesis tests (McNemar, Wilcoxon, Holm)
├── generate_report_artifacts.py# Generator for publication figures and CSV tables
├── scores.csv                  # 405 verified RAGAS evaluation scores
├── runs/
│   ├── test.jsonl              # Original raw log; contains repeated question IDs
│   └── test_unique.jsonl       # 405 unique condition records (135 questions × 3)
├── report_artifacts/           # Final publication figures and summary tables
│   ├── figure_1_main_metrics.png
│   ├── figure_2_latency_distribution.png
│   ├── main_metrics.csv
│   ├── efficiency_metrics.csv
│   └── efficiency_comparisons.csv
├── analysis_output/            # Statistical test results and hypothesis decisions
│   ├── statistical_comparisons.csv
│   └── hypothesis_decisions.csv
├── submission/                 # Corrected manuscript, dossier, slides, and PDFs
├── prepare_unique_runs.py      # Builds the paired de-duplicated runtime log
├── error_analysis_completed.csv# Preliminary qualitative sample (35 rows)
├── Final.ipynb                  # Interactive GPU experiment runner for Google Colab
├── requirements.txt            # Python environment dependencies
└── LICENSE                     # MIT Open-Source License
```

---

## 🛠️ Installation & Independent Run

```bash
# Clone the repository
git clone https://github.com/Baknur1/noise-filtering.git
cd noise-filtering

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run full reproduction pipeline
python reproduce_all.py
```

---

## 📜 Academic Integrity and Data Availability Statement

The repository preserves the original runtime log and adds a de-duplicated paired log for analysis. The saved-data reproduction was checked locally. Independent reproduction by another person, a fresh end-to-end model run, author approval, and a university similarity report are not claimed by the repository.
Licensed under the [MIT License](LICENSE).
