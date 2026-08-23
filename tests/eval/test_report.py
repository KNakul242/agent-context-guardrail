import pytest

from src.eval.report import build_eval_report

# Same fixture as tests/eval/test_metrics.py, plus a hard-negative mask --
# indices 2, 4, 5, 7 (the benign ones) are treated as hard negatives here.
Y_TRUE = [1, 1, 0, 1, 0, 0, 1, 0]
Y_SCORES = [0.6, 0.5, 0.4, 0.35, 0.3, 0.2, 0.15, 0.1]
IS_HARD_NEGATIVE = [False, False, True, False, True, True, False, True]


def test_report_contains_every_dod_required_metric():
    """docs/specs/DEVELOPMENT_RULES.md DoD / IMPLEMENTATION_PLAN.md Phase 1:
    F1, ROC-AUC, recall@1%-FPR, and FPR on the hard-negative benign subset
    specifically -- all four, not a subset of them."""
    report = build_eval_report(Y_TRUE, Y_SCORES, IS_HARD_NEGATIVE, threshold=0.35)
    assert set(report.keys()) >= {
        "f1", "roc_auc", "recall_at_1pct_fpr", "hard_negative_fpr", "n_examples", "n_hard_negative",
        "hard_negative_fpr_numerator", "hard_negative_fpr_denominator",
    }


def test_report_counts_match_input_size():
    report = build_eval_report(Y_TRUE, Y_SCORES, IS_HARD_NEGATIVE, threshold=0.35)
    assert report["n_examples"] == 8
    assert report["n_hard_negative"] == 4


def test_report_hard_negative_fpr_uses_only_the_hard_negative_subset():
    """Threshold 0.35: predictions are [1,1,1,1,0,0,0,0]. Of the 4
    hard-negative rows (indices 2,4,5,7 -> true labels 0,0,0,0), predicted
    labels are [1,0,0,0] -> 1 false positive out of 4 = 0.25. If this
    function wrongly used the full 8-row set instead of just the
    hard-negative subset, this would not match."""
    report = build_eval_report(Y_TRUE, Y_SCORES, IS_HARD_NEGATIVE, threshold=0.35)
    assert report["hard_negative_fpr"] == pytest.approx(0.25)


def test_report_hard_negative_fpr_numerator_and_denominator_match_the_rate():
    """ds-review LOW-MEDIUM finding: the bare percentage invites reporting
    e.g. "3% FPR" without the small-sample caveat -- numerator/denominator
    must travel alongside it so a caller can't drop that context."""
    report = build_eval_report(Y_TRUE, Y_SCORES, IS_HARD_NEGATIVE, threshold=0.35)
    assert report["hard_negative_fpr_denominator"] == 4
    assert report["hard_negative_fpr_numerator"] == 1
    assert report["hard_negative_fpr_numerator"] / report["hard_negative_fpr_denominator"] == pytest.approx(
        report["hard_negative_fpr"]
    )


def test_report_recall_at_1pct_fpr_matches_the_metrics_module_directly():
    from src.eval.metrics import recall_at_fpr

    report = build_eval_report(Y_TRUE, Y_SCORES, IS_HARD_NEGATIVE, threshold=0.35)
    assert report["recall_at_1pct_fpr"] == pytest.approx(recall_at_fpr(Y_TRUE, Y_SCORES, max_fpr=0.01))


def test_report_raises_a_clear_error_when_no_hard_negatives_present():
    with pytest.raises(ValueError):
        build_eval_report([1, 0], [0.9, 0.1], [False, False], threshold=0.5)
