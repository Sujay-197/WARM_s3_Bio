"""GO-based rule scoring following the paper's B1-B6 approximation."""

from collections import Counter

import pandas as pd

from .rule_mining import Rule


def score_rule(
    matrix: pd.DataFrame,
    rule: Rule,
    annotations: dict[str, list[str]],
    epsilon: float = 0.001,
) -> pd.DataFrame:
    """Score rows containing the antecedent but not the consequent.

    Annotation keys are transaction labels (matrix row labels). The returned frame
    includes the candidate label, confidence, weighted score, and GO coverage.
    """
    antecedent_rows = matrix.loc[:, list(rule.antecedent)].all(axis=1)
    consequent_rows = matrix.loc[:, list(rule.consequent)].all(axis=1)
    training_rows = matrix.index[antecedent_rows & consequent_rows]
    candidate_rows = matrix.index[antecedent_rows & ~consequent_rows]
    training_terms = Counter(
        term for row in training_rows for term in annotations.get(row, [])
    )
    training_count = len(training_rows)
    records = []
    for row in candidate_rows:
        terms = set(annotations.get(row, []))
        probabilities = [
            training_terms[term] / training_count
            for term in terms
            if training_count and training_terms[term]
        ]
        score = rule.confidence
        for probability in probabilities:
            score *= probability
        score *= epsilon ** max(0, len(terms) - len(probabilities))
        records.append(
            {"transaction": row, "confidence": rule.confidence, "score": score,
             "observed_terms": len(probabilities), "candidate_terms": len(terms)}
        )
    return pd.DataFrame.from_records(records, columns=[
        "transaction", "confidence", "score", "observed_terms", "candidate_terms"
    ])


def score_rules(
    matrix: pd.DataFrame, rules: list[Rule], annotations: dict[str, list[str]], epsilon: float = 0.001
) -> pd.DataFrame:
    frames = []
    for rule in rules:
        scored = score_rule(matrix, rule, annotations, epsilon)
        if not scored.empty:
            scored["antecedent"] = [rule.antecedent] * len(scored)
            scored["consequent"] = [rule.consequent] * len(scored)
            frames.append(scored)
    if not frames:
        return pd.DataFrame(columns=["transaction", "confidence", "score", "observed_terms", "candidate_terms", "antecedent", "consequent"])
    return pd.concat(frames, ignore_index=True)
