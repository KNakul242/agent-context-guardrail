import pytest

from src.eval.metrics import (
    bypass_rate,
    f1,
    false_positive_rate,
    recall_at_fpr,
    roc_auc,
    segment_by_category,
)

# Hand-checked against sklearn.metrics.roc_curve directly (see docs/DECISIONS.md
# context around this module for the derivation) -- not just asserting
# "whatever sklearn returns," so this test actually catches a wrong
# threshold-selection rule in recall_at_fpr, not just a broken import.
SCORES = [0.6, 0.5, 0.4, 0.35, 0.3, 0.2, 0.15, 0.1]
LABELS = [1, 1, 0, 1, 0, 0, 1, 0]


def test_recall_at_fpr_zero_percent():
    # Only the two highest scores (both positive) achieve FPR == 0.
    assert recall_at_fpr(LABELS, SCORES, max_fpr=0.01) == pytest.approx(0.5)


def test_recall_at_fpr_wider_budget():
    # At FPR <= 0.3, the best achievable point is FPR=0.25 / recall=0.75.
    assert recall_at_fpr(LABELS, SCORES, max_fpr=0.3) == pytest.approx(0.75)


def test_recall_at_fpr_full_budget():
    assert recall_at_fpr(LABELS, SCORES, max_fpr=1.0) == pytest.approx(1.0)


def test_f1_perfect_predictions():
    assert f1(y_true=[0, 1, 1, 0], y_pred=[0, 1, 1, 0]) == pytest.approx(1.0)


def test_f1_all_wrong():
    assert f1(y_true=[0, 1, 1, 0], y_pred=[1, 0, 0, 1]) == pytest.approx(0.0)


def test_roc_auc_perfect_separation():
    assert roc_auc(y_true=[0, 0, 1, 1], y_scores=[0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_false_positive_rate_on_pure_negative_subset():
    # 3 of 4 hard-negative benign examples wrongly flagged -- this is the
    # exact shape docs/specs/DEVELOPMENT_RULES.md's DoD calls "FPR on the
    # hard-negative benign subset."
    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 1, 0]
    assert false_positive_rate(y_true, y_pred) == pytest.approx(0.75)


def test_bypass_rate_on_pure_positive_subset():
    # 1 of 4 malicious examples slipped through undetected (predicted benign).
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 0]
    assert bypass_rate(y_true, y_pred) == pytest.approx(0.25)


def test_segment_by_category_reports_per_category_not_aggregate():
    """D10: red-team/eval results must be segmented by category, never
    collapsed into one aggregate number."""
    y_true = [1, 1, 1, 1]
    y_pred = [0, 1, 0, 1]  # bypassed, caught, bypassed, caught
    categories = ["encoding", "encoding", "direct_override", "direct_override"]

    result = segment_by_category(y_true, y_pred, categories, metric_fn=bypass_rate)

    assert result == {
        "encoding": pytest.approx(0.5),
        "direct_override": pytest.approx(0.5),
    }


def test_segment_by_category_does_not_silently_average_across_categories():
    y_true = [1, 1, 1]
    y_pred = [0, 0, 1]
    categories = ["a", "a", "b"]

    result = segment_by_category(y_true, y_pred, categories, metric_fn=bypass_rate)

    assert result["a"] == pytest.approx(1.0)
    assert result["b"] == pytest.approx(0.0)
