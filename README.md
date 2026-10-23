# WARL: HCV-Human Protein Interaction Prediction

WARL is a research-oriented Python project for predicting candidate interactions between Hepatitis C virus (HCV) proteins and human proteins. It combines interaction-network pattern mining with Gene Ontology (GO) molecular-function evidence.

The output is a ranked list of computational hypotheses for further biological review. It does not experimentally prove that an interaction occurs.

## Problem statement

HCV survives and replicates by interacting with proteins inside human cells. Known HCV-human protein-interaction databases are incomplete, while laboratory experiments to discover additional interactions are expensive and slow.

> Given known HCV-human protein interactions, which unrecorded human-HCV protein pairs are plausible candidates for further investigation?

## Dataset

Input file: `data/raw/virhostnet_hcv_human_interactions.csv`

| Item | Count |
| --- | ---: |
| Known interactions | 591 |
| Human proteins | 527 |
| HCV proteins | 10 |

The HCV proteins are `CORE`, `E1`, `E2`, `p7`, `NS2`, `NS3`, `NS4A`, `NS4B`, `NS5A`, and `NS5B`.

Each input row contains one known pair:

```text
human_protein,viral_protein
A2RUB6,NS3
O00468,CORE
```

An absent pair is not automatically a biological negative. It may simply be an interaction that has not yet been recorded in this dataset.

## Method overview

```text
Known interaction CSV
        |
        v
HV and VH binary matrices
        |
        v
BiMax closed-itemset biclustering
        |
        v
Itemset expansion and association rules
        |
        v
GO molecular-function weighting
        |
        v
Ranked candidate HCV-human interactions
```

### 1. Matrix construction

The edge list is converted into a binary HV matrix:

- Rows: human proteins
- Columns: HCV proteins
- `1`: the interaction is recorded
- `0`: the interaction is not recorded in the input dataset

`VH` is the transpose of this matrix. Running the method in both directions helps identify patterns from both the human-protein and viral-protein viewpoints.

### 2. BiMax biclustering

BiMax finds closed all-1 submatrices, also called closed itemsets or biclusters. These are recurring interaction patterns: a group of human proteins that shares a group of viral interaction partners.

### 3. Association rules

The project expands useful patterns into small itemsets and produces association rules such as:

```text
{V1} -> {V2}
```

The rule confidence is:

```text
confidence(A -> C) = support(A union C) / support(A)
```

High confidence means the pattern repeatedly occurs in the observed interaction network. It does not establish biological causality.

### 4. GO molecular-function weighting

UniProt provides GO molecular-function annotations for human proteins. WARL uses these annotations to add biological context to association rules.

The score combines rule confidence, GO-term co-occurrence evidence, and a missing-term penalty (`epsilon = 0.001`). This helps rank candidates that are both network-consistent and functionally compatible with the proteins supporting a rule.

## Current output

Output file: `data/predictions.csv`

| Check | Result |
| --- | ---: |
| Prediction rows | 1,862 |
| Unique `(human, viral)` pairs | 1,862 |
| Duplicate pairs | 0 |
| HV-derived predictions | 1,836 |
| VH-derived predictions | 26 |
| Overlap with the 591 input pairs | 0 |

All 1,900 predictions are novel **relative to the input CSV**. This does not mean that every pair is unknown in every external database or experimentally unverified.

| Output column | Meaning |
| --- | --- |
| `human` | Candidate human protein identifier |
| `viral` | Candidate HCV protein |
| `weighted_confidence` | GO-weighted confidence used to rank the prediction |
| `unweighted_confidence` | Raw association-rule confidence before GO weighting |
| `direction` | Matrix direction that produced the prediction (`HV` or `VH`) |

## Installation

From PowerShell in the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
```

Expected result:

```text
15 passed
```

## Run the pipeline

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py `
  data\raw\virhostnet_hcv_human_interactions.csv `
  > data\predictions.csv
```

The first full run may contact UniProt to fetch GO annotations. Responses are cached at `data/cache/go_annotations.json`; the cache is intentionally ignored by Git.

Inspect the top predictions in PowerShell:

```powershell
Import-Csv data\predictions.csv |
  Sort-Object {[double]$_.weighted_confidence} -Descending |
  Select-Object -First 20 human,viral,weighted_confidence,unweighted_confidence,direction |
  Format-Table
```

Open the CSV in the default spreadsheet application:

```powershell
Invoke-Item data\predictions.csv
```

## Evaluation and baseline comparison

Run the evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_warm.py
```

Evaluation protocol:

- Fixed random seed: `42`
- Training interactions: 474
- Held-out known positives: 117
- Sampled provisional negatives: 117
- Held-out interactions are removed before training to avoid direct protein-pair leakage.

The sampled negatives are unobserved pairs, not confirmed biological non-interactions.

| Model | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| WARM | 0.500 | 1.000 | 0.667 | 0.665 |
| XGBoost baseline | 0.500 | 1.000 | 0.667 | 0.393 |

The implemented WARM run achieved a higher PR-AUC than the XGBoost baseline. The XGBoost baseline uses graph-derived features including human-protein degree, viral-protein degree, common-neighbor count, and neighborhood overlap.

## Repository layout

```text
data/
  raw/                 Source interaction CSV
  predictions.csv      Generated ranked candidates
src/warl/
  data_loading.py      CSV loading and HV/VH matrix construction
  bimax.py             Closed-itemset biclustering
  itemset_expansion.py Candidate subset generation
  rule_mining.py        Support, confidence, and rule generation
  go_annotations.py    UniProt GO fetching and on-disk cache
  go_weighting.py      GO-aware scoring
  pipeline.py          End-to-end workflow
  evaluation.py        Evaluation helpers
scripts/
  run_pipeline.py      Command-line pipeline entry point
  evaluate_warm.py     Held-out WARM and XGBoost evaluation
tests/                 Automated test suite
output/pdf/            Generated project report
```

## Limitations

- Predictions are computational hypotheses, not laboratory-confirmed interactions.
- The dataset is VirHostNet-derived and differs from the historical HCVpro dataset used in the source paper.
- GO annotations and external databases change over time.
- Candidate expansion uses a deterministic cap for very large biclusters to keep the computation tractable.
- The evaluation's unobserved pairs are provisional negatives, so metrics should be interpreted carefully.
- Stronger validation would compare predictions with an independent interaction database and, ultimately, laboratory experiments.

## Reports

- `output/pdf/WARL_project_report.pdf` - project summary, methodology, results, and limitations
- `output/pdf/WARL_theory_guide.pdf` - short theory guide
- `output/pdf/WARL_beginner_theory_guide.pdf` - detailed beginner-first explanation

## Reference

Indhumathy M, Nabhan AR, Arumugam S. *A Weighted Association Rule Mining Method for Predicting HCV-Human Protein Interactions.* Current Bioinformatics. 2018;13:73-84.
