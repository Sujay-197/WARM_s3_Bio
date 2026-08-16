# WARL: Weighted Association Rule Mining for HCV-Human PPI Prediction — Design

## Context

Implementation of the method from:

> Indhumathy M, Nabhan AR, Arumugam S. "A Weighted Association Rule Mining Method for
> Predicting HCV-Human Protein Interactions." *Current Bioinformatics* 2018;13:73-84.
> (`ref_hcv.pdf`, full text extracted to `ref_hcv.txt`)

Course project (not production code): prioritize readability and explainability over
performance. Author is an ML/AI engineer with no prior bio domain background — bio-specific
steps are explained inline in code/notebooks, not assumed knowledge.

## Dataset

- Source: HCVpro database export (manually curated HCV-human PPI database), TSV/CSV with
  HCV protein ID and human protein ID/UniProt ID columns. Provided by the user.
- Raw: 11 HCV proteins, 455 human proteins, 549 interactions → 524 after de-duplication.
- GO molecular function annotations fetched per protein via UniProt REST API
  (`https://rest.uniprot.org/uniprotkb/{id}`), cached to disk.

## Method (from the paper)

1. **Matrix construction**: bipartite edge list → `HV` binary incidence matrix (rows =
   human proteins, cols = viral proteins, 1 = interaction), `VH = HV.T`.
2. **BiMax biclustering**: applied to both `HV` and `VH` to extract *closed itemsets*
   (bicluster column-sets where no proper superset shares the same support/row-count).
3. **Itemset expansion**: raw closed itemsets from BiMax on this sparse dataset have low
   support / high cardinality. For each closed itemset `S`, generate permutation-subsets
   `S'` (max size 4). Keep `S'` only if `support(S') > support(S)`.
4. **Rule generation**: for itemset `S` (`|S|=n`), for each item `i ∈ S`: consequent
   `C={i}`, antecedent `A=S-{i}` (Eq. 1). `Conf(A⇒C) = Support(A∪C)/Support(A)`
   (Appendix A).
5. **GO-based rule weighting** (core novelty):
   - `P(C,T|A) ∝ P(C|A)·P(T|A)` by Bayes + conditional independence (Eq. 2/B1).
   - `P(C|A) = Conf(A⇒C)`.
   - `M(T) = {t1,...,tn}`: GO molecular-function terms of the protein represented by
     transaction `T` (Eq. 3/B2).
   - `P(T|A) ≈ P(M(T)|A) = ∏ P(ti|A)` assuming conditional independence of terms given
     `A` (Eq. B3/B4).
   - `P(ti|A)`: estimated by counting how often GO term `t` co-occurs with the training
     rows (rows containing both `A` and `C`), normalized to a probability.
   - **Penalty factor**: for a candidate transaction observing only `m` of the `n` GO
     terms seen in training, multiply by `ε^(n-m)`, `ε = 0.001` (constant from the
     paper). `n=m` ⇒ no penalty.
   - Final score: `P(C,M(T)|A) = P(C|A) · ∏_{t∈f} P(t|A) · ε^(n-|f|)` (Eq. B6).
6. **Threshold filtering**: paper's empirical cutoff on `P(C,M(T)|A)` was `1e-14`.
   Configurable in our implementation — coverage differences vs. the 2018 UniProt/GO
   snapshot mean we expect to retune.
7. **Rule deduplication**: combine predictions from `HV`-derived and `VH`-derived rules;
   drop exact-duplicate `(human, viral)` pairs and rules whose antecedent is a subset of
   an already-covered pair. Paper reduced 30 (HV) + 46 (VH) → 42 unique interactions.
8. **Evaluation**: sanity-check predictions aren't already-known interactions; lightweight
   GO-term frequency grouping per predicted viral hub protein (in place of the paper's
   DAVID/REVIGO enrichment step — no external tool dependency).

## Architecture

```
WARM_s3_Bio/
├── src/warl/
│   ├── data_loading.py       # HCVpro export -> edge list -> HV/VH incidence matrices
│   ├── go_annotations.py     # UniProt REST client + on-disk cache for GO-MF terms
│   ├── bimax.py               # custom BiMax biclustering (closed itemsets)
│   ├── itemset_expansion.py  # subset expansion for sparse closed itemsets
│   ├── rule_mining.py        # rule generation + support/confidence (Eq 1, Appendix A)
│   ├── go_weighting.py       # P(go|A), penalty factor, P(C,M(T)|A) (Eq 2-5, B1-B6)
│   ├── pipeline.py           # orchestrates HV+VH runs, dedup/merge, final ranking
│   ├── evaluation.py         # known-interaction cross-check, GO-term frequency grouping
│   └── config.py             # eps=0.001, threshold=1e-14, max subset size=4 (overridable)
├── notebooks/
│   └── 01_explore.ipynb      # mirrors the src/ steps for interactive exploration
├── scripts/run_pipeline.py   # CLI entry point, end-to-end
├── data/
│   ├── raw/                  # HCVpro export (gitignored: data/cache, data/processed only)
│   ├── cache/                 # cached GO annotations (JSON, keyed by protein ID) [gitignored]
│   └── processed/             # generated matrices, intermediate itemsets [gitignored]
├── tests/
└── docs/superpowers/specs/   # this file
```

Modules take/return plain pandas DataFrames/dicts — no shared global state — so each step
can be run and inspected independently in the notebook.

## Data flow

1. `data_loading.py`: parse edges, dedupe (549→524), build `HV`/`VH`.
2. `bimax.py`: recursive divide-and-conquer BiMax (Prelić et al. formulation) over the
   binary matrix — small enough (455×11) that a plain-Python implementation is fast and
   fully inspectable; no dependency on R/rpy2 or sklearn's non-closed-itemset biclustering.
3. `itemset_expansion.py`: permutation-subset expansion, max size 4, support-filtered.
4. `rule_mining.py`: rule generation per Eq. 1 + support/confidence per Appendix A.
5. `go_annotations.py`: one UniProt REST call per protein ID, cached to
   `data/cache/go_annotations.json`; proteins with zero returned GO terms keep an empty
   term set (handled naturally by the penalty factor, not an error case).
6. `go_weighting.py`, per rule `A⇒C`: build `P(t|A)` from training rows (contain `A` and
   `C`), score candidate rows (contain `A`, not `C`) via Eq. B6.
7. `pipeline.py`: run steps 2-6 independently on `HV` and `VH`, threshold-filter (default
   `1e-14`, also reports the "average of training scores" cutoff as a diagnostic), merge
   and dedup per the paper's Section 3.2 logic.
8. `evaluation.py`: known-interaction cross-check; GO-term frequency counts (`Counter`/
   groupby) per predicted viral hub protein.

## Testing

- **Golden test from the paper**: Appendix A's Table A1 5×4 toy matrix has a worked
  answer (`{V1}⇒{V3}` confidence = 0.75) — encoded as a fixture; `rule_mining.py` must
  reproduce it exactly. Same table doubles as a BiMax smoke test.
- **Penalty factor**: unit test Eq. B6 against hand-computed values for small `n`, `m`.
- **GO client**: UniProt HTTP calls mocked in the test suite (no live network in tests);
  live fetching is a manual/notebook step.

## Deliverables

1. Modular Python pipeline (`src/warl/`, one concern per module).
2. `notebooks/01_explore.ipynb` for exploration + `scripts/run_pipeline.py` for a full run.
3. Ranked list of predicted new HCV-human interactions with confidence + weighted score.
4. Brief comparison against the paper's reported 42 predictions (overlap count; dataset/
   API differences vs. 2018 are expected).

## Phasing

Implementation proceeds phase by phase, pausing for review between phases:

- **Phase 1 (current)**: `data_loading.py`, `go_annotations.py`, `bimax.py` — matrix
  construction, GO fetching/caching, and biclustering. Blocked on the user providing the
  HCVpro export file for `data_loading.py`; `go_annotations.py` and `bimax.py` can be
  built and tested (with the paper's toy example) independently of the real dataset.
- **Phase 2+ (later)**: `itemset_expansion.py`, `rule_mining.py`, `go_weighting.py`,
  `pipeline.py`, `evaluation.py` — deferred until Phase 1 is reviewed.
