"""
In-distribution eval report assembly (Phase 1 DoD, docs/specs/
IMPLEMENTATION_PLAN.md): F1, ROC-AUC, recall@1%-FPR (mirrors PromptGuard 2's
own headline metric, D6), and FPR on the hard-negative benign subset
specifically -- all four together, since scripts/evaluate.py is the only
place they need to be assembled and none of them should be reported without
the others (a lone F1 or a lone hard-negative FPR each tell only part of the
story this DoD item asks for).

Kept as a pure function over plain lists (no model, no I/O) so it's testable
without a checkpoint -- scripts/evaluate.py is the thin, untested wiring
layer that turns a real model + dataset split into these same lists.
"""

from typing import Sequence

from src.eval.metrics import f1, false_positive_rate, recall_at_fpr, roc_auc


def build_eval_report(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    is_hard_negative: Sequence[bool],
    threshold: float = 0.5,
) -> dict:
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    hard_true = [t for t, h in zip(y_true, is_hard_negative) if h]
    hard_pred = [p for p, h in zip(y_pred, is_hard_negative) if h]
    hard_negative_false_positives = sum(1 for t, p in zip(hard_true, hard_pred) if t == 0 and p == 1)

    return {
        "n_examples": len(y_true),
        "n_hard_negative": len(hard_true),
        "threshold": threshold,
        "f1": f1(y_true, y_pred),
        "roc_auc": roc_auc(y_true, y_scores),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, y_scores, max_fpr=0.01),
        # ds-review LOW-MEDIUM finding: NotInject's per-split hard-negative
        # count is small (~30-35 rows in val/test) -- a bare percentage
        # invites reporting "3% FPR" without the tiny denominator that makes
        # a single false positive worth ~3 points. Numerator/denominator
        # travel with the rate specifically so a caller can't drop that
        # context downstream.
        "hard_negative_fpr": false_positive_rate(hard_true, hard_pred),
        "hard_negative_fpr_numerator": hard_negative_false_positives,
        "hard_negative_fpr_denominator": len(hard_true),
    }
