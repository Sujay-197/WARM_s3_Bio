"""End-to-end WARL pipeline for HCV-human interaction prediction."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .bimax import bimax
from .data_loading import build_hv_matrix, build_vh_matrix, load_interactions
from .go_annotations import GOAnnotationCache, fetch_go_molecular_function
from .go_weighting import score_rules
from .itemset_expansion import expand_itemsets
from .rule_mining import generate_rules


def _run_direction(matrix: pd.DataFrame, annotations: dict[str, list[str]], max_itemset_size: int, annotation_mode: str):
    clusters = bimax(matrix)
    itemsets = expand_itemsets(matrix, clusters, max_itemset_size)
    rules = generate_rules(matrix, itemsets)
    return score_rules(matrix, rules, annotations, annotation_mode=annotation_mode)


def run_pipeline(
    interaction_path: str,
    cache_path: str = "data/cache/go_annotations.json",
    threshold: float = 1e-14,
    max_itemset_size: int = 4,
) -> pd.DataFrame:
    edges = load_interactions(interaction_path)
    hv = build_hv_matrix(edges)
    vh = build_vh_matrix(hv)
    cache = GOAnnotationCache(cache_path)
    labels = set(hv.index) | set(hv.columns)
    missing = [label for label in labels if label not in cache._cache]
    # UniProt is a network service; parallel requests make the real dataset practical.
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(fetch_go_molecular_function, label, None, 3): label for label in missing}
        for future in as_completed(futures):
            label = futures[future]
            try:
                cache._cache[label] = future.result()
            except Exception:
                # Missing/unavailable annotations are handled by the paper's penalty model.
                cache._cache[label] = []
    if missing:
        cache._save()
    annotations = {label: cache._cache.get(label, []) for label in labels}
    predictions = []
    for matrix, direction, annotation_mode in ((hv, "HV", "row"), (vh, "VH", "consequent")):
        scored = _run_direction(matrix, annotations, max_itemset_size, annotation_mode)
        if scored.empty:
            continue
        scored = scored[scored["weighted_confidence"] >= threshold].copy()
        scored["direction"] = direction
        predictions.append(scored)
    if not predictions:
        return pd.DataFrame(columns=["human", "viral", "weighted_confidence", "unweighted_confidence", "direction"])
    result = pd.concat(predictions, ignore_index=True)
    result["human"] = result.apply(
        lambda row: row["transaction"] if row["direction"] == "HV" else row["consequent"].__iter__().__next__(), axis=1
    )
    result["viral"] = result.apply(
        lambda row: row["consequent"].__iter__().__next__() if row["direction"] == "HV" else row["transaction"], axis=1
    )
    return result.sort_values("weighted_confidence", ascending=False).drop_duplicates(["human", "viral"])
