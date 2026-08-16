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
