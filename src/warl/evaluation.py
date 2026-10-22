"""Small evaluation helpers for predicted protein interactions."""

import pandas as pd


def known_interactions(predictions: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    known = set(zip(edges["human"], edges["viral"]))
    return predictions.apply(lambda row: (row["human"], row["viral"]) in known, axis=1)


def go_term_counts(predictions: pd.DataFrame, annotations: dict[str, list[str]]) -> pd.Series:
    terms = [term for protein in predictions["human"] for term in annotations.get(protein, [])]
    return pd.Series(terms, dtype="object").value_counts()
