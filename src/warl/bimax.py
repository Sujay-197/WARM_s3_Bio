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
