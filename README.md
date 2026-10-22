# WARL: HCV-Human Protein Interaction Prediction

This project implements a readable research pipeline inspired by weighted association-rule mining for predicting HCV-human protein interactions.

## What it does

1. Loads the VirHostNet-derived HCV-human interaction dataset.
2. Builds human-by-viral (`HV`) and viral-by-human (`VH`) binary matrices.
3. Finds closed interaction patterns with custom BiMax biclustering.
4. Expands patterns into itemsets and association rules.
5. Fetches and caches UniProt Gene Ontology molecular-function annotations.
6. Scores and ranks candidate interactions using confidence and GO evidence.

## Setup

From PowerShell at the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
```

## Run the pipeline

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py `
  data\raw\virhostnet_hcv_human_interactions.csv `
  > data\predictions.csv
```

UniProt responses are cached under `data/cache/`, which is intentionally ignored by Git. The current generated prediction output is stored in `data/predictions.csv`.

## Report

The generated project report is available at `output/pdf/WARL_project_report.pdf`.

## Repository layout

- `src/warl/` - pipeline modules
- `tests/` - automated tests
- `data/raw/` - source interaction dataset
- `data/predictions.csv` - ranked prediction output
- `output/pdf/` - project report
- `scripts/` - pipeline and report commands

The predictions are computational hypotheses, not experimentally confirmed interactions. The report documents the dataset differences from the original paper and the current implementation limitations.
