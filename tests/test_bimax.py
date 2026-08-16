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
