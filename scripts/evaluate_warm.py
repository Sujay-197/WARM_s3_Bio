"""Leakage-aware held-out evaluation for the WARM scorer.

Known edges are split by viral protein with a deterministic seed. An equal number
of unobserved pairs are sampled as provisional negatives; these are evaluation
labels, not claims that the biology is negative.
"""

from pathlib import Path
import random
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from warl.pipeline import run_pipeline


def pr_auc(y, scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    positives = sum(y)
    if not positives:
        return 0.0
    tp = fp = previous_recall = area = 0.0
    for i in order:
        if y[i]: tp += 1
        else: fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/raw/virhostnet_hcv_human_interactions.csv"
    edges = pd.read_csv(source).drop_duplicates()
    rng = random.Random(42)
    test_indices = []
    for _, group in edges.groupby("viral_protein"):
        count = max(1, round(len(group) * 0.2))
        test_indices.extend(rng.sample(list(group.index), count))
    test = edges.loc[test_indices].copy()
    train = edges.drop(index=test_indices).copy()
    (root / "tmp").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir=root / "tmp") as handle:
        train.to_csv(handle.name, index=False)
        train_path = Path(handle.name)
    try:
        predictions = run_pipeline(str(train_path), str(root / "data/cache/go_annotations.json"))
    finally:
        train_path.unlink(missing_ok=True)

    humans = sorted(edges.human_protein.unique())
    virals = sorted(edges.viral_protein.unique())
    known = set(zip(train.human_protein, train.viral_protein))
    positives = set(zip(test.human_protein, test.viral_protein))
    candidates = [(h, v) for h in humans for v in virals if (h, v) not in known and (h, v) not in positives]
    negatives = rng.sample(candidates, min(len(candidates), len(positives)))
    labels = [1] * len(positives) + [0] * len(negatives)
    score_map = {(r.human, r.viral): r.score for r in predictions.itertuples()}
    scores = [score_map.get(pair, 0.0) for pair in positives] + [score_map.get(pair, 0.0) for pair in negatives]
    rows = []
    for threshold in sorted(set(scores), reverse=True):
        predicted = [score >= threshold for score in scores]
        tp = sum(y and p for y, p in zip(labels, predicted))
        fp = sum((not y) and p for y, p in zip(labels, predicted))
        fn = sum(y and not p for y, p in zip(labels, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((f1, threshold, precision, recall))
    best = max(rows) if rows else (0.0, 0.0, 0.0, 0.0)
    result = {
        "seed": 42, "train_edges": len(train), "test_positive_edges": len(positives),
        "sampled_negative_edges": len(negatives), "precision": best[2], "recall": best[3],
        "f1": best[0], "pr_auc": pr_auc(labels, scores), "threshold": best[1],
        "note": "Negatives are unobserved pairs sampled for evaluation and are not confirmed biological negatives.",
    }
    print(pd.Series(result).to_json(indent=2))


if __name__ == "__main__":
    main()
