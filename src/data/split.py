"""
Stratified, document-group-aware train/val/test split
(docs/specs/IMPLEMENTATION_PLAN.md Phase 0 DoD: "train/val/test split,
seeded"; docs/DECISIONS.md D24/D25).

Stratifies by (source, label) at the document-group level, not the raw-row
level -- that distinction is the whole point after D25. BIPIA's benign and
malicious rows are a matched-pair design: D25 found that 100% of curated
BIPIA malicious rows share their underlying clean document with a benign
row (same content, before vs. after attack-string insertion). A plain
per-row (source, label) split would let that same document's benign form
land in train while a malicious variant of it lands in test -- the model
could then partly recognize the *document*, not the injection, and the eval
split would leak signal it's supposed to be measuring generalization
against.

document_group_key() groups every BIPIA row derived from the same
clean_context_id (parsed from the notes field src/data/sources/bipia.py
already writes -- both its "clean" and "poisoned" branches include
task=...; clean_context_id=... verbatim) into one unit; everything else
(NotInject, prodnull, MAlmasabi -- no matched-pair structure, each row is
independent) is its own singleton group, so those sources split exactly as
before this change.
"""

import random
import re
from collections import defaultdict
from typing import List, Tuple

from src.data.schema import Example

_BIPIA_DOC_KEY_RE = re.compile(r"task=(\w+).*clean_context_id=(\d+)")


def document_group_key(ex: Example):
    """The unit that must move together through a split. For BIPIA, this
    is the underlying document (task, clean_context_id) shared by a clean
    row and every malicious row derived from it (D25). For every other
    source, a row has no such family -- its own example_id is a group of
    one, which is equivalent to no grouping at all."""
    if ex.source == "bipia":
        m = _BIPIA_DOC_KEY_RE.search(ex.notes)
        if m:
            return ("bipia_doc", m.group(1), m.group(2))
    return ("singleton", ex.example_id)


def _group_stratum_key(group: List[Example]):
    """Which (source, label)-style bucket a whole document group counts
    under, for stratification quotas. A BIPIA document family mixes labels
    by construction (one clean + N poisoned) -- "mixed" is a real, distinct
    stratum, not a fallback; every BIPIA family currently falls in it."""
    sources = {ex.source for ex in group}
    assert len(sources) == 1, f"a document group must not span multiple sources, got {sources}"
    source = sources.pop()
    labels = {ex.label for ex in group}
    if len(labels) > 1:
        return (source, "mixed")
    return (source, labels.pop())


def stratified_split(
    examples: List[Example],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> Tuple[List[Example], List[Example], List[Example]]:
    assert abs(sum(ratios) - 1.0) < 1e-9, f"ratios must sum to 1.0, got {ratios}"
    train_ratio, val_ratio, _test_ratio = ratios

    doc_groups = defaultdict(list)
    for ex in examples:
        doc_groups[document_group_key(ex)].append(ex)

    strata = defaultdict(list)  # stratum -> list of document groups (each a list of Examples)
    for group in doc_groups.values():
        strata[_group_stratum_key(group)].append(group)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for group_list in strata.values():
        shuffled = group_list[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        # round() per boundary rather than floor: a stratum of e.g. 3
        # groups at 80/10/10 should still put its majority in train, not
        # lose it to rounding down twice in a row.
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        for group in shuffled[:n_train]:
            train.extend(group)
        for group in shuffled[n_train : n_train + n_val]:
            val.extend(group)
        for group in shuffled[n_train + n_val :]:
            test.extend(group)

    return train, val, test
