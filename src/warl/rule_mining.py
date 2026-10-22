"""Association-rule generation and support/confidence calculations."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Rule:
    antecedent: frozenset
    consequent: frozenset
    support: int
    confidence: float


def support(matrix: pd.DataFrame, items: frozenset) -> int:
    if not items:
        return len(matrix)
    return int(matrix.loc[:, list(items)].all(axis=1).sum())


def generate_rules(matrix: pd.DataFrame, itemsets: list[frozenset]) -> list[Rule]:
    rules = []
    for itemset in itemsets:
        if len(itemset) < 2:
            continue
        joint = support(matrix, itemset)
        for item in sorted(itemset, key=str):
            consequent = frozenset({item})
            antecedent = itemset - consequent
            antecedent_support = support(matrix, antecedent)
            confidence = joint / antecedent_support if antecedent_support else 0.0
            rules.append(Rule(antecedent, consequent, joint, confidence))
    return rules
