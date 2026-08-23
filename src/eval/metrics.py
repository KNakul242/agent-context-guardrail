"""
Eval metric functions (docs/specs/DEVELOPMENT_RULES.md TDD scope; D10's
per-category segmentation requirement).

f1() and roc_auc() are thin wrappers, not reinventions -- sklearn already
implements them correctly, so the only reason they live here is to pin one
convention (positive label = 1 = Label.MALICIOUS) everywhere they're called
from, rather than leaving each call site to decide pos_label/average
independently and risk drifting apart.

recall_at_fpr() is the one genuinely custom metric: PromptGuard 2's own
headline number is "recall at 1% FPR" (D6's ablation is designed for direct
comparability against it), and sklearn has no ready-made function for it.
"""

from typing import Callable, Sequence

from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return f1_score(y_true, y_pred, pos_label=1)


def roc_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    return roc_auc_score(y_true, y_scores)


def recall_at_fpr(y_true: Sequence[int], y_scores: Sequence[float], max_fpr: float) -> float:
    """Best recall (TPR) achievable at any threshold whose FPR does not
    exceed max_fpr. roc_curve() already returns (fpr, tpr) pairs sorted by
    non-increasing threshold with fpr non-decreasing, so the qualifying
    points are a prefix of the arrays -- take the max tpr among them
    (tpr is not itself monotonic within that prefix: a lower threshold can
    add a true positive without adding a false positive, e.g. the 0.35
    threshold in this module's test case)."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    qualifying_tpr = tpr[fpr <= max_fpr]
    return float(qualifying_tpr.max())


def threshold_at_fpr(y_true: Sequence[int], y_scores: Sequence[float], max_fpr: float) -> float:
    """The actual score cutoff that achieves recall_at_fpr's reported
    recall -- recall_at_fpr() answers "how good is the best achievable
    point," this answers "what threshold gets you there." Needed because
    scripts/evaluate.py's headline recall@1%-FPR number and
    scripts/run_redteam.py's --threshold defaulted to two unrelated
    decision boundaries (0.5 has no connection to the 1%-FPR operating
    point) -- a red-team bypass rate measured at the wrong threshold
    answers a different question than the eval report's headline metric,
    which matters directly for D7's ModernBERT gate (peer-review finding).
    Same qualifying-prefix logic as recall_at_fpr, returning the threshold
    at the point of max tpr within that prefix instead of the tpr itself."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    qualifying = fpr <= max_fpr
    qualifying_tpr = tpr[qualifying]
    qualifying_thresholds = thresholds[qualifying]
    best_index = qualifying_tpr.argmax()
    return float(qualifying_thresholds[best_index])


def false_positive_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """FP / (FP + TN). Called with a hard-negative-only subset's
    (y_true, y_pred) to get docs/specs/DEVELOPMENT_RULES.md's "FPR on the
    hard-negative benign subset" -- this function itself doesn't know or
    care what the subset is, that's the caller's job (D10: never bake a
    specific segmentation into a metric function; segment_by_category()
    below is the general mechanism)."""
    negatives = [(t, p) for t, p in zip(y_true, y_pred) if t == 0]
    if not negatives:
        raise ValueError("false_positive_rate: no negative examples in y_true")
    false_positives = sum(1 for t, p in negatives if p == 1)
    return false_positives / len(negatives)


def bypass_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """FN / (FN + TP): fraction of actually-malicious examples the detector
    missed. This is red-team ASR from the detector's point of view -- the
    metric Phase 1's "bypass rate by InjectionTechnique category" (D10)
    reports, via segment_by_category() below."""
    positives = [(t, p) for t, p in zip(y_true, y_pred) if t == 1]
    if not positives:
        raise ValueError("bypass_rate: no positive examples in y_true")
    false_negatives = sum(1 for t, p in positives if p == 0)
    return false_negatives / len(positives)


def segment_by_category(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    categories: Sequence[str],
    metric_fn: Callable[[Sequence[int], Sequence[int]], float],
) -> dict:
    """Apply metric_fn separately within each category group -- the general
    mechanism behind D10's rule that eval/red-team results are reported per
    `InjectionTechnique` category, never collapsed into one aggregate
    number. Deliberately returns a dict keyed by category with no combined
    entry: computing an aggregate here would make it too easy for a caller
    to grab the one number D10 says not to report."""
    grouped_true: dict = {}
    grouped_pred: dict = {}
    for t, p, c in zip(y_true, y_pred, categories):
        grouped_true.setdefault(c, []).append(t)
        grouped_pred.setdefault(c, []).append(p)

    return {c: metric_fn(grouped_true[c], grouped_pred[c]) for c in grouped_true}
