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
