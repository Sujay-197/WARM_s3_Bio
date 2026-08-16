"""One-off builder for notebooks/00_academic_report.ipynb.

Run this whenever the academic-report notebook's content needs regenerating:
    .venv/Scripts/python.exe scripts/build_academic_notebook.py

This script is the *source of truth* for the notebook's cells; the .ipynb file is a
generated, executed artifact (like a compiled report), not hand-edited.
"""

import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md(
    """\
# WARL: Weighted Association Rule Mining for HCV-Human PPI Prediction

**Academic report notebook** — records the methodology, data provenance, and decisions
behind this implementation, separate from `01_explore.ipynb`'s step-by-step pipeline
walkthrough. Regenerate this notebook with `scripts/build_academic_notebook.py` rather
than editing it by hand.

Implementation of the method from:

> Indhumathy M, Nabhan AR, Arumugam S. "A Weighted Association Rule Mining Method for
> Predicting HCV-Human Protein Interactions." *Current Bioinformatics* 2018;13:73-84.

Full spec: [`docs/superpowers/specs/2026-08-16-warl-hcv-ppi-design.md`](../docs/superpowers/specs/2026-08-16-warl-hcv-ppi-design.md)"""
)

md(
    """\
## 1. Background

Hepatitis C virus (HCV) causes chronic liver disease. Understanding which human
proteins HCV's own proteins physically interact with — a *protein-protein interaction*
(PPI) network — helps explain how the virus hijacks host-cell machinery. Experimentally
mapping every possible interaction is slow and expensive, so this project predicts
*new*, previously unobserved interactions computationally from a set of already-known
ones, using **weighted association rule mining (WARL)**.

The core data structure is a **bipartite graph**: two disjoint sets of nodes (11 HCV
proteins in the paper's case, hundreds of human proteins), with edges only ever running
between the two sets, never within one (no human-human or viral-viral edges). An edge
means "these two proteins are known to interact."

**Gene Ontology (GO)** is a controlled vocabulary describing what a protein *does*.
This project uses only its *molecular function* branch (e.g. "protein binding", "serine
protease activity") — one of GO's three sub-ontologies (the others being biological
process and cellular component) — as side information to make association rules more
confident even when raw interaction data is sparse (Section 5 below)."""
)

md(
    """\
## 2. Data source and provenance

### 2.1 The paper's original source

- **Source**: HCVpro (Kwofie et al. 2011) — manually curated, aggregating literature +
  VirusMint + VirHostNet + BIND.
- **Format**: tab-separated text (TSV); the original export also carried GO/pathway/OMIM
  annotation columns alongside the interactions themselves.
- **Raw**: 11 HCV proteins, 455 human proteins, 549 interactions.
- **After dedup**: 524 interactions, 466 total nodes → a very sparse bipartite graph
  (524 edges across 466 possible endpoints).
- GO molecular function annotations were used only for rule *weighting*, not as raw
  mining features.

### 2.2 HCVpro is no longer reachable

Both known HCVpro mirrors were checked directly (as of 2026-08-16) and are down:

| Mirror | Result |
|---|---|
| `apps.sanbi.ac.za/hcvpro` | DNS resolution fails entirely — domain is gone |
| `cbrc.kaust.edu.sa/hcvpro` | Repeated HTTP 503/504 — server unreachable |

The Wayback Machine's full crawl history for `apps.sanbi.ac.za/hcvpro` was pulled via
its CDX API. It only ever archived the static UI shell (the search form, FAQ, and the
HTML of the download page) — the dynamic `download_helper.php` endpoint that actually
served the interaction file was never crawled, so the original dataset is not
recoverable from the archive.

### 2.3 Substitution: VirHostNet

**VirHostNet** is itself one of HCVpro's original source databases, and unlike HCVpro it
is still actively maintained (now on a "3.0-beta" release). Its documented PSICQUIC
web-service endpoint (`virhostnet.prabi.fr:9090`) turned out to be dead too — confirmed
independently two ways: a direct connection attempt gets `Connection refused` at the TCP
level, and the official EBI PSICQUIC registry lists VirHostNet's entry as
`active: false`, `count: 0`.

The live path that does work: VirHostNet's current network is published on **NDEx**
(Network Data Exchange) as `VirHostNet-v3.0321`
(UUID `9dd36e92-994e-11eb-9e72-0ac135e8bacf`) — 10,049 nodes, 40,005 edges, public,
confirmed downloadable.

**Extraction steps applied to that network:**

1. Downloaded the full CX2-format network export (~18 MB).
2. Identified genuine HCV strain nodes by NCBI taxonomy ID: **Con1** (taxid `333284`)
   and **JFH-1** (taxid `356411`). A third candidate match on substring `"HCV"` (taxid
   `868565`) turned out to be a false positive — Human Herpesvirus 8 — and was excluded.
3. Each strain's proteins are recorded as UniProt PRO-cleavage-product IDs of a single
   polyprotein accession (Con1 = `Q9WMX2`, JFH-1 = `Q99IB8`). Fetched each polyprotein's
   UniProt entry and read its `Chain` features to map PRO IDs to the paper's 10 named
   proteins (CORE, E1, E2, p7, NS2, NS3, NS4A, NS4B, NS5A, NS5B).
4. Interactions attributed only to the *uncleaved* polyprotein (no cleavage-site detail
   in the curation) were dropped, per the simplest-viable-mapping decision made for this
   project.
5. Filtered edges to human (taxid `9606`) ↔ HCV pairs, pooled across both strains, and
   deduplicated.

### 2.4 Stats comparison

| | Paper (HCVpro, 2018) | This implementation (VirHostNet, 2026) |
|---|---|---|
| Interactions | 524 | 591 |
| Human proteins | 455 | 527 |
| Viral proteins | 11 | 10 |

The paper's 11th protein, **F** (a.k.a. ARFP), arises from a ribosomal frameshift during
translation rather than simple proteolytic cleavage of the main polyprotein, so it is
never annotated as a UniProt `Chain` feature and is absent from this extraction."""
)

code(
    """\
import sys
sys.path.insert(0, "../src")

from warl.data_loading import load_interactions, build_hv_matrix, build_vh_matrix

edges = load_interactions("../data/raw/virhostnet_hcv_human_interactions.csv")
print(f"Loaded {len(edges)} deduplicated interactions")
edges.head()"""
)

md(
    """\
## 3. Matrix construction

Following the paper's Section 2.1: the bipartite graph is represented as two binary
incidence matrices. **HV** has human proteins as rows and viral proteins as columns
(1 = known interaction); **VH** is simply its transpose. Both matrices feed into BiMax
biclustering independently (Section 2.4 of the paper), since the two orientations
surface different closed itemsets."""
)

code(
    """\
hv = build_hv_matrix(edges)
vh = build_vh_matrix(hv)

print(f"HV matrix: {hv.shape[0]} human proteins x {hv.shape[1]} viral proteins")
print(f"VH matrix: {vh.shape[0]} viral proteins x {vh.shape[1]} human proteins")
print(f"\\nViral proteins: {list(hv.columns)}")
print(f"Matrix density: {hv.values.sum()} / {hv.size} cells filled "
      f"({100 * hv.values.sum() / hv.size:.2f}%) -- confirms the sparsity the paper "
      f"describes as motivating the itemset-expansion step (Section 2.3.1)")"""
)

code(
    """\
hv.head(10)"""
)

md(
    """\
## 4. Method summary (for reference)

Full equations and derivations are in the paper (`ref_hcv.pdf`) and the design spec.
Kept here as a compact reference:

1. **BiMax biclustering** extracts *closed itemsets* from `HV`/`VH`: a set of columns is
   closed if no proper superset shares its support (row count) — equivalently, an
   inclusion-maximal all-1s submatrix.
2. **Itemset expansion**: closed itemsets from sparse data tend to have low support and
   high cardinality. Subsets (permutations, max size 4) with *higher* support than their
   parent itemset are added to the rule-generation pool.
3. **Rule generation**: for itemset `S` (`|S| = n`), each item `i ∈ S` becomes a
   consequent `C = {i}` with antecedent `A = S - {i}`; confidence
   `Conf(A⇒C) = Support(A∪C) / Support(A)`.
4. **GO-based weighting** (Eq. 2-5, B1-B6): rescales each rule's confidence using how
   well a candidate protein's GO molecular-function terms match the terms observed among
   proteins that already support the rule, with a penalty factor `ε^(n-m)` (`ε = 0.001`)
   for candidates missing GO annotations that training proteins had.
5. **Threshold filtering & dedup**: keep predictions above a score cutoff (paper used
   `1e-14`, empirically tuned), merge HV- and VH-derived rules, drop duplicate/subsumed
   predictions.

*(Sections 5-6 — BiMax results, GO weighting, final predictions, evaluation — will be
appended here once Phase 2 is implemented.)*"""
)

nb["cells"] = cells

client = NotebookClient(nb, timeout=120, kernel_name="warl-venv", resources={"metadata": {"path": "notebooks"}})
client.execute()

with open("notebooks/00_academic_report.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote and executed notebooks/00_academic_report.ipynb")
