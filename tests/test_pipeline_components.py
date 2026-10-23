import pandas as pd

from warl.go_weighting import score_rule
from warl.itemset_expansion import expand_itemsets
from warl.rule_mining import Rule, generate_rules, support
from warl.bimax import Bicluster


def _matrix():
    return pd.DataFrame({"V1": [1, 1, 0], "V2": [1, 0, 1], "V3": [0, 1, 1]}, index=["H1", "H2", "H3"])


def test_support_and_rule_generation():
    matrix = _matrix()
    assert support(matrix, frozenset({"V1"})) == 2
    rules = generate_rules(matrix, [frozenset({"V1", "V2"})])
    assert len(rules) == 2
    assert all(rule.support == 1 for rule in rules)


def test_itemset_expansion_keeps_support_increasing_subsets():
    matrix = _matrix()
    cluster = Bicluster(frozenset({"H1"}), frozenset({"V1", "V2"}))
    assert expand_itemsets(matrix, [cluster], max_size=2) == [frozenset({"V1"}), frozenset({"V2"})]


def test_go_weighting_scores_candidate_with_penalty():
    matrix = _matrix()
    rule = Rule(frozenset({"V1"}), frozenset({"V2"}), 1, 0.5)
    scored = score_rule(matrix, rule, {"H1": ["binding"], "H2": ["binding"], "H3": ["other"]})
    assert list(scored["transaction"]) == ["H2"]
    assert scored.iloc[0]["unweighted_confidence"] == 0.5
    assert scored.iloc[0]["weighted_confidence"] == 0.5


def test_go_weighting_penalizes_candidate_without_annotations():
    matrix = _matrix()
    rule = Rule(frozenset({"V1"}), frozenset({"V2"}), 1, 0.5)
    scored = score_rule(matrix, rule, {"H1": ["binding"], "H2": [], "H3": []})
    assert scored.iloc[0]["unweighted_confidence"] == 0.5
    assert scored.iloc[0]["weighted_confidence"] < 0.5
