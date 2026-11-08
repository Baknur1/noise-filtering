# Assignment 4 Reproducibility Package

This repository evaluates whether post-retrieval context filtering changes hallucination rate and answer correctness in a technical-documentation RAG system. The experiment compares three paired conditions on the same RAGBench TechQA questions: dense Top-10 retrieval, a frozen Top-4 length-matched control, and a cross-encoder-filtered condition.

## Current status

The package contains a completed three-question validation smoke generation run. It does **not** yet contain the final test run, RAGAS score files, statistical output, figures, manual error analysis, or an independent reproduction record. Therefore, no final hypothesis conclusion can currently be reproduced from this package.

## Files

- `rag_filter_pipeline.py` retrieves contexts, applies the experimental conditions, calls the local Ollama generator, and writes traceable JSONL records.
- `evaluate_ragas.py` calculates per-question RAGAS metrics with a separate local judge model.
- `paired_analysis.py` performs paired McNemar and Wilcoxon analyses with bootstrap confidence intervals.
- `consolidated_analysis.py` runs all three pairwise comparisons, applies Holm correction across declared families, and writes machine-readable hypothesis decisions.
- `generate_report_artifacts.py` creates the main metric table, efficiency table, and two required figures from saved files.
- `validate_final_package.py` checks the 200-question and 600-record evidence gates before report finalisation.
- `Appendix_A_freeze.md` records the frozen design.
- `RUNBOOK.md` gives the Windows PowerShell execution order.
- `runs/` stores raw generation logs.
- `logs/` stores statistical command output.

## Prerequisites

- Python 3.12
- Ollama running locally
- generator model `llama3.1:8b`
- independent judge model `qwen2.5:7b`
- enough disk space for Hugging Face datasets and model caches

Install dependencies in a fresh virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

## Validation smoke test

Do not tune on the test split.

```powershell
python rag_filter_pipeline.py --split validation --n 3 --tau -2.0 --control-m 4 --out runs\smoke_new.jsonl
python evaluate_ragas.py runs\smoke_new.jsonl runs\smoke_scores.csv --judge qwen2.5:7b
```

Check that there are 9 scored rows, three conditions per question, and no unexplained `nan` values.

## Frozen final run

Run only after the protocol has been frozen and its Moodle upload time has been preserved.

```powershell
python rag_filter_pipeline.py --split test --n 200 --tau -2.0 --control-m 4 --out runs\test.jsonl
python evaluate_ragas.py runs\test.jsonl scores.csv --judge qwen2.5:7b
python paired_analysis.py scores.csv baseline filtered | Tee-Object logs\stats_baseline_vs_filtered.txt
python paired_analysis.py scores.csv length_matched filtered | Tee-Object logs\stats_control_vs_filtered.txt
python paired_analysis.py scores.csv baseline length_matched | Tee-Object logs\stats_baseline_vs_control.txt
python consolidated_analysis.py scores.csv report_artifacts
python generate_report_artifacts.py scores.csv runs\test.jsonl report_artifacts
```

The commands are resumable. Preserve failed attempts and document any deviation from the frozen protocol.

After completing the manual error-analysis CSV, run the final machine checks:

```powershell
python validate_final_package.py --run runs\test.jsonl --scores scores.csv --errors error_analysis_completed.csv
```

## Integrity checks

Each generation log begins with a manifest containing the split, sample size, dataset fingerprint, frozen parameters, and SHA-256 hash of the sorted question IDs. Every reported number must trace back to `test.jsonl`, `scores.csv`, or a saved analysis output.

## Reproduction test

Before submission, a person outside the implementation process must run the documented commands in a clean environment and complete Appendix C. Record the operating system, machine, commit, release tag, commands, differences, and any README fixes.

## Repository release checklist

- Select and add an appropriate `LICENSE` file.
- Remove secrets, local caches, and model files.
- Commit raw logs needed to trace the report, subject to dataset licensing.
- Create the Git tag `assignment-4`.
- Add the final repository URL to the report.
