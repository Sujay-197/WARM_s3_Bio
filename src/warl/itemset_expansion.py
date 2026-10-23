"""Expand sparse closed biclusters into support-increasing itemsets."""

from itertools import combinations, islice

import pandas as pd

from .bimax import Bicluster


def expand_itemsets(
    matrix: pd.DataFrame, biclusters: list[Bicluster], max_size: int = 4,
    max_candidates_per_cluster: int = 1000,
) -> list[frozenset]:
    """Return unique column subsets whose support exceeds their parent support."""
    result: set[frozenset] = set()
    for bicluster in biclusters:
        columns = sorted(bicluster.columns, key=str)
        for size in range(1, min(max_size, len(columns)) + 1):
            for combo in islice(combinations(columns, size), max_candidates_per_cluster):
                itemset = frozenset(combo)
                support = int((matrix.loc[:, list(itemset)].all(axis=1)).sum())
                if support > bicluster.support:
                    result.add(itemset)
    return sorted(result, key=lambda s: (len(s), tuple(map(str, sorted(s)))))
