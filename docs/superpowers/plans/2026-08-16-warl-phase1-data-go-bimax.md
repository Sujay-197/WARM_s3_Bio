# WARL Phase 1 (data_loading, go_annotations, bimax) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test the first three pipeline modules — HCVpro edge-list loading and
matrix construction, UniProt GO molecular-function fetching with an on-disk cache, and a
custom BiMax closed-itemset (biclustering) implementation — each independently testable
without needing the real HCVpro export file.

**Architecture:** Three standalone modules under `src/warl/`, each pure-function/small-class
style, taking and returning plain pandas objects or lists/dicts. No module depends on
another in this phase. Tests use the paper's own worked example (Appendix A, Table A1 — a
5×4 toy matrix with a documented confidence value) as a golden fixture, and a fake
`requests.Session` for the UniProt client so the test suite never touches the network.

**Tech Stack:** Python 3.14 (already installed at `C:\Python314`), pandas 3.0.5, requests
2.34.2, pytest 9.1.1, in a project-local `.venv`. Full design context, the paper's
equations, and later phases are in
[docs/superpowers/specs/2026-08-16-warl-hcv-ppi-design.md](../specs/2026-08-16-warl-hcv-ppi-design.md).

## Global Constraints

- Course project: prioritize readability/explainability over performance (no premature
  optimization, no defensive code for scenarios that can't happen).
- No live network calls in the automated test suite — the UniProt client is tested via a
  fake `requests.Session`.
- Package layout is `src/warl/`; tests resolve it via `pyproject.toml`'s
  `[tool.pytest.ini_options] pythonpath = ["src"]` (already configured and verified working
  — no package install/`pip install -e .` needed).
- `.venv/` holds pandas 3.0.5, requests 2.34.2, pytest 9.1.1 (already created; see Task 1
  Step 1 to verify, don't recreate).
- Run all commands from the repo root: `C:\Users\work\work\clg\bio`.
- Test runner: `.venv/Scripts/python.exe -m pytest -v`.

---

### Task 1: Project scaffolding + `data_loading.py`

**Files:**
- Verify (already created, untracked): `pyproject.toml`, `requirements.txt`,
  `src/warl/__init__.py` (empty), `tests/__init__.py` (empty)
- Create: `src/warl/data_loading.py`
- Test: `tests/test_data_loading.py`

**Interfaces:**
- Produces:
  - `load_interactions(path: str, human_col: str = "human_protein", viral_col: str = "viral_protein") -> pd.DataFrame` — two columns named exactly `"human"` and `"viral"`, deduplicated.
  - `build_hv_matrix(edges: pd.DataFrame) -> pd.DataFrame` — rows = human protein labels, columns = viral protein labels, values `{0, 1}` (`int`).
  - `build_vh_matrix(hv: pd.DataFrame) -> pd.DataFrame` — `hv.T`.

- [ ] **Step 1: Verify the existing scaffolding**

Run:
```bash
cd "C:\Users\work\work\clg\bio"
".venv/Scripts/python.exe" -m pytest -v
```
Expected: `collected 0 items` / `no tests ran in 0.00s` (pytest exits with code 5 for an
empty test suite — that's expected here, not a failure). Confirms the venv and
`pyproject.toml` `pythonpath` config resolve correctly before writing any test.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_data_loading.py`:

```python
import pandas as pd

from warl.data_loading import build_hv_matrix, build_vh_matrix, load_interactions

TABLE_A1_CSV = """human_protein,viral_protein
H1,V1
H1,V1
H1,V2
H2,V1
H2,V3
H3,V4
H4,V1
H4,V3
H5,V1
H5,V2
H5,V3
H5,V4
"""


def test_load_interactions_dedupes_rows(tmp_path):
    csv_path = tmp_path / "interactions.csv"
    csv_path.write_text(TABLE_A1_CSV)

    edges = load_interactions(str(csv_path))

    assert list(edges.columns) == ["human", "viral"]
    assert len(edges) == 11  # the H1,V1 duplicate row is removed
    assert not edges.duplicated().any()


def test_load_interactions_renames_custom_columns(tmp_path):
    csv_path = tmp_path / "interactions.csv"
    csv_path.write_text("hcv_id,human_id\nV1,H1\nV2,H1\n")

    edges = load_interactions(str(csv_path), human_col="human_id", viral_col="hcv_id")

    assert list(edges.columns) == ["human", "viral"]
    assert len(edges) == 2


def test_build_hv_matrix_matches_table_a1(tmp_path):
    csv_path = tmp_path / "interactions.csv"
    csv_path.write_text(TABLE_A1_CSV)
    edges = load_interactions(str(csv_path))

    hv = build_hv_matrix(edges)

    assert hv.shape == (5, 4)
    assert hv.loc["H1", "V1"] == 1
    assert hv.loc["H1", "V3"] == 0
    assert hv.loc["H5"].sum() == 4
    assert hv.loc["H3", "V4"] == 1


def test_build_vh_matrix_is_transpose_of_hv(tmp_path):
    csv_path = tmp_path / "interactions.csv"
    csv_path.write_text(TABLE_A1_CSV)
    edges = load_interactions(str(csv_path))
    hv = build_hv_matrix(edges)

    vh = build_vh_matrix(hv)

    assert vh.shape == (4, 5)
    assert vh.loc["V1", "H1"] == hv.loc["H1", "V1"]
    assert vh.loc["V4", "H3"] == hv.loc["H3", "V4"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_data_loading.py -v`
Expected: `ModuleNotFoundError: No module named 'warl.data_loading'` (or `ImportError`) for
all 4 tests.

- [ ] **Step 4: Write the implementation**

Create `src/warl/data_loading.py`:

```python
"""Load HCVpro interaction exports and build HV/VH incidence matrices."""

import pandas as pd


def load_interactions(
    path: str, human_col: str = "human_protein", viral_col: str = "viral_protein"
) -> pd.DataFrame:
    """Load an HCVpro-style interaction export and return a deduplicated edge list.

    Returns a DataFrame with exactly two columns, "human" and "viral".
    """
    raw = pd.read_csv(path, sep=None, engine="python")
    edges = raw.rename(columns={human_col: "human", viral_col: "viral"})[["human", "viral"]]
    return edges.drop_duplicates().reset_index(drop=True)


def build_hv_matrix(edges: pd.DataFrame) -> pd.DataFrame:
    """Build the HV binary incidence matrix: rows = human proteins, columns = viral proteins."""
    hv = pd.crosstab(edges["human"], edges["viral"])
    return (hv > 0).astype(int)


def build_vh_matrix(hv: pd.DataFrame) -> pd.DataFrame:
    """VH is the transpose of HV: rows = viral proteins, columns = human proteins."""
    return hv.T
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_data_loading.py -v`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt src/warl/__init__.py tests/__init__.py src/warl/data_loading.py tests/test_data_loading.py .gitignore
git commit -m "feat: add project scaffolding and data_loading module"
```

---

### Task 2: `go_annotations.py`

**Files:**
- Create: `src/warl/go_annotations.py`
- Test: `tests/test_go_annotations.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `fetch_go_molecular_function(accession: str, session: requests.Session | None = None, timeout: int = 10) -> list[str]` — list of GO molecular-function term *names* (the `F:`-prefixed UniProt GO terms with the prefix stripped), empty list if the accession has no entry (HTTP 404) or no MF cross-references.
  - `GOAnnotationCache(cache_path: str)` with method
    `get_or_fetch(accession: str, session: requests.Session | None = None) -> list[str]` —
    returns cached terms if present, otherwise calls `fetch_go_molecular_function` and
    persists the full cache to `cache_path` as JSON after every new fetch.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_go_annotations.py`:

```python
import json

from warl.go_annotations import GOAnnotationCache, fetch_go_molecular_function


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return self._response


UNIPROT_PAYLOAD = {
    "uniProtKBCrossReferences": [
        {
            "database": "GO",
            "id": "GO:0005515",
            "properties": [
                {"key": "GoTerm", "value": "F:protein binding"},
                {"key": "GoEvidenceType", "value": "IEA:InterPro"},
            ],
        },
        {
            "database": "GO",
            "id": "GO:0008270",
            "properties": [{"key": "GoTerm", "value": "F:zinc ion binding"}],
        },
        {
            "database": "GO",
            "id": "GO:0006351",
            "properties": [{"key": "GoTerm", "value": "P:transcription, DNA-templated"}],
        },
        {
            "database": "InterPro",
            "id": "IPR000123",
            "properties": [{"key": "EntryName", "value": "Some_domain"}],
        },
    ]
}


def test_fetch_go_molecular_function_extracts_only_mf_terms():
    session = _FakeSession(_FakeResponse(200, UNIPROT_PAYLOAD))

    terms = fetch_go_molecular_function("P12345", session=session)

    assert terms == ["protein binding", "zinc ion binding"]
    assert session.calls == ["https://rest.uniprot.org/uniprotkb/P12345.json"]


def test_fetch_go_molecular_function_returns_empty_on_404():
    session = _FakeSession(_FakeResponse(404))

    terms = fetch_go_molecular_function("NOTAREALID", session=session)

    assert terms == []


def test_cache_fetches_once_and_persists_to_disk(tmp_path):
    cache_path = tmp_path / "go_cache.json"
    session = _FakeSession(_FakeResponse(200, UNIPROT_PAYLOAD))
    cache = GOAnnotationCache(str(cache_path))

    first = cache.get_or_fetch("P12345", session=session)
    second = cache.get_or_fetch("P12345", session=session)

    assert first == ["protein binding", "zinc ion binding"]
    assert second == first
    assert len(session.calls) == 1  # second call served from the in-memory cache

    with open(cache_path) as f:
        on_disk = json.load(f)
    assert on_disk == {"P12345": ["protein binding", "zinc ion binding"]}


def test_cache_loads_existing_file_on_init(tmp_path):
    cache_path = tmp_path / "go_cache.json"
    cache_path.write_text(json.dumps({"P99999": ["kinase activity"]}))
    cache = GOAnnotationCache(str(cache_path))

    terms = cache.get_or_fetch("P99999", session=_FakeSession(_FakeResponse(500)))

    assert terms == ["kinase activity"]  # served from disk, never touched the (failing) session
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_go_annotations.py -v`
Expected: `ModuleNotFoundError: No module named 'warl.go_annotations'` for all 4 tests.

- [ ] **Step 3: Write the implementation**

Create `src/warl/go_annotations.py`:

```python
"""Fetch and cache Gene Ontology molecular-function annotations from UniProt."""

import json
import os

import requests

UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"


def fetch_go_molecular_function(
    accession: str, session: requests.Session | None = None, timeout: int = 10
) -> list[str]:
    """Fetch GO molecular-function term names for a single UniProt accession.

    Returns an empty list if the accession has no entry or no molecular-function
    GO cross-references -- proteins with sparse annotations are a normal part of
    this dataset, not an error case.
    """
    session = session or requests.Session()
    response = session.get(UNIPROT_ENTRY_URL.format(accession=accession), timeout=timeout)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    entry = response.json()

    terms = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "GO":
            continue
        for prop in xref.get("properties", []):
            if prop.get("key") == "GoTerm" and prop.get("value", "").startswith("F:"):
                terms.append(prop["value"][2:])
    return terms


class GOAnnotationCache:
    """On-disk JSON cache mapping a UniProt accession to its GO-MF term names."""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache: dict[str, list[str]] = json.load(f)
        else:
            self._cache = {}

    def get_or_fetch(
        self, accession: str, session: requests.Session | None = None
    ) -> list[str]:
        if accession not in self._cache:
            self._cache[accession] = fetch_go_molecular_function(accession, session=session)
            self._save()
        return self._cache[accession]

    def _save(self) -> None:
        directory = os.path.dirname(self.cache_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_go_annotations.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/warl/go_annotations.py tests/test_go_annotations.py
git commit -m "feat: add UniProt GO molecular-function fetching with disk cache"
```

---

### Task 3: `bimax.py`

**Files:**
- Create: `src/warl/bimax.py`
- Test: `tests/test_bimax.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2 (takes any 0/1 `pd.DataFrame`, e.g. Task 1's
  `build_hv_matrix`/`build_vh_matrix` output, but isn't coupled to it).
- Produces:
  - `Bicluster` frozen dataclass: fields `rows: frozenset[str]`, `columns: frozenset[str]`;
    property `support: int` (= `len(rows)`).
  - `bimax(matrix: pd.DataFrame, min_support: int = 1) -> list[Bicluster]` — all closed
    itemsets (BiMax biclusters) in `matrix`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bimax.py`:

```python
import pandas as pd

from warl.bimax import bimax


def _table_a1_matrix() -> pd.DataFrame:
    data = {
        "V1": [1, 1, 0, 1, 1],
        "V2": [1, 0, 0, 0, 1],
        "V3": [0, 1, 0, 1, 1],
        "V4": [0, 0, 1, 0, 1],
    }
    return pd.DataFrame(data, index=["H1", "H2", "H3", "H4", "H5"])


def test_bimax_finds_all_closed_itemsets_in_table_a1():
    matrix = _table_a1_matrix()

    biclusters = bimax(matrix)

    found = {(bc.rows, bc.columns) for bc in biclusters}
    expected = {
        (frozenset({"H1", "H2", "H4", "H5"}), frozenset({"V1"})),
        (frozenset({"H1", "H5"}), frozenset({"V1", "V2"})),
        (frozenset({"H2", "H4", "H5"}), frozenset({"V1", "V3"})),
        (frozenset({"H3", "H5"}), frozenset({"V4"})),
        (frozenset({"H5"}), frozenset({"V1", "V2", "V3", "V4"})),
    }
    assert found == expected


def test_bimax_support_equals_row_count():
    matrix = _table_a1_matrix()

    biclusters = bimax(matrix)

    support_by_columns = {bc.columns: bc.support for bc in biclusters}
    assert support_by_columns[frozenset({"V1"})] == 4
    assert support_by_columns[frozenset({"V1", "V2", "V3", "V4"})] == 1


def test_bimax_min_support_filters_low_support_biclusters():
    matrix = _table_a1_matrix()

    biclusters = bimax(matrix, min_support=2)

    assert all(bc.support >= 2 for bc in biclusters)
    columns_found = {bc.columns for bc in biclusters}
    assert frozenset({"V1", "V2", "V3", "V4"}) not in columns_found  # support 1, filtered out


def test_bimax_on_transposed_matrix_produces_dual_biclusters():
    vh = _table_a1_matrix().T  # rows = viral proteins, columns = human proteins

    biclusters = bimax(vh)

    dual = {(bc.rows, bc.columns) for bc in biclusters}
    assert (frozenset({"V1"}), frozenset({"H1", "H2", "H4", "H5"})) in dual
```

Note on the expected set in the first test: it's derived by hand from
`_table_a1_matrix()` using the paper's closed-itemset definition (a set of columns is
closed if no proper superset shares its support/row-set) — see
`docs/superpowers/specs/2026-08-16-warl-hcv-ppi-design.md` for the full worked trace if
the assertion needs re-deriving later.

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_bimax.py -v`
Expected: `ModuleNotFoundError: No module named 'warl.bimax'` for all 4 tests.

- [ ] **Step 3: Write the implementation**

Create `src/warl/bimax.py`:

```python
"""BiMax biclustering: extraction of closed itemsets from a binary incidence matrix.

A closed itemset is a set of columns whose set of supporting rows (its extent) is
not shared by any proper superset of columns -- equivalently, (rows, columns) is an
inclusion-maximal all-1s submatrix. This brute-forces over subsets of whichever axis
(rows or columns) is smaller and closes each subset across the row/column Galois
connection -- exact and fast here because one axis of both the HV and VH matrices is
always the 11 HCV proteins.
"""

from dataclasses import dataclass
from itertools import combinations

import pandas as pd


@dataclass(frozen=True)
class Bicluster:
    rows: frozenset
    columns: frozenset

    @property
    def support(self) -> int:
        return len(self.rows)


def _itemsets(matrix: pd.DataFrame, transpose: bool) -> dict[str, frozenset]:
    """Map each label on one axis to the frozenset of labels on the other axis
    where the matrix has a 1."""
    frame = matrix.T if transpose else matrix
    return {label: frozenset(frame.columns[frame.loc[label] == 1]) for label in frame.index}


def bimax(matrix: pd.DataFrame, min_support: int = 1) -> list[Bicluster]:
    """Find all closed itemsets (BiMax biclusters) in a binary incidence matrix.

    `matrix` must contain only 0/1 values; rows and columns can be any hashable
    labels (e.g. protein IDs).
    """
    by_row = _itemsets(matrix, transpose=False)  # row label -> set of columns with a 1
    by_col = _itemsets(matrix, transpose=True)   # column label -> set of rows with a 1

    columns_are_smaller = len(matrix.columns) <= len(matrix.index)
    if columns_are_smaller:
        seed_labels = list(matrix.columns)
        first_lookup, second_lookup = by_col, by_row
    else:
        seed_labels = list(matrix.index)
        first_lookup, second_lookup = by_row, by_col

    seen = set()
    biclusters: list[Bicluster] = []

    for size in range(1, len(seed_labels) + 1):
        for combo in combinations(seed_labels, size):
            other_axis = frozenset.intersection(*(first_lookup[label] for label in combo))
            if not other_axis:
                continue
            seed_axis = frozenset.intersection(*(second_lookup[label] for label in other_axis))
            if not seed_axis:
                continue

            if columns_are_smaller:
                rows, columns = other_axis, seed_axis
            else:
                rows, columns = seed_axis, other_axis

            if len(rows) < min_support:
                continue

            key = (rows, columns)
            if key in seen:
                continue
            seen.add(key)
            biclusters.append(Bicluster(rows=rows, columns=columns))

    return biclusters
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_bimax.py -v`
Expected: `4 passed`

- [ ] **Step 5: Run the full test suite**

Run: `".venv/Scripts/python.exe" -m pytest -v`
Expected: `12 passed` (4 from Task 1 + 4 from Task 2 + 4 from Task 3)

- [ ] **Step 6: Commit**

```bash
git add src/warl/bimax.py tests/test_bimax.py
git commit -m "feat: add custom BiMax closed-itemset biclustering"
```

---

## After Phase 1

Phase 1 delivers matrix construction, GO annotation fetching, and biclustering — all
independently tested. Per the design doc's phasing, **stop here** and pause for review.
Phase 2 (`itemset_expansion.py`, `rule_mining.py`, `go_weighting.py`, `pipeline.py`,
`evaluation.py`) is deferred to a separate plan, and `data_loading.py`'s real-data path
is still blocked on the user providing the actual HCVpro export file — Task 1's tests
cover the loader's logic via the paper's toy example in the meantime.
