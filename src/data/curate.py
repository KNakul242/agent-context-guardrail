"""
Composition-pass building blocks (docs/DECISIONS.md D13 resolution;
docs/data_summary.md).

Reusable pieces the actual curation run (see the bottom of this file's
module, or the report this accompanies) is built from — kept separate and
unit-testable because getting the sampling logic wrong here silently biases
the training set in a way nothing downstream would catch.
"""

import random
import re
from collections import defaultdict
from typing import Callable, List


def stratified_cap(rows: List[dict], key_fn: Callable[[dict], object], cap: int, seed: int = 0) -> List[dict]:
    """Group rows by key_fn(row), keep up to `cap` per group (all of it if
    the group is smaller). Used to downsample BIPIA's malicious combinatorial
    volume to its actual template/domain/position diversity (D13 resolution)
    without touching groups that are already at or below the cap."""
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)

    rng = random.Random(seed)
    result = []
    for group in groups.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        result.extend(shuffled[:cap])
    return result


# Structural markers for the 3 BIPIA domains we have local content for.
# Anything matching none of these is "unclassified" -- in practice this
# bucket is dominated by BIPIA's excluded WebQA/Summarization domains
# (docs/data_summary.md, XSum/NewsQA finding), not a catch-all for noise.
def classify_domain(context: str) -> str:
    if re.search(r"^\s*\|.+\|.*\n\s*\|[-\s|:]+\|", context, re.MULTILINE) or context.count("|") > 10:
        return "table"
    if re.search(r"SUBJECT:|EMAIL_FROM:|RECEIVED DATE:|From:.*@|Subject:", context):
        return "email"
    if re.search(r"```|Traceback|def \w+\(|import \w+|SyntaxError|ValueError|TypeError", context):
        return "code"
    return "unclassified"


def entity_names(s: str) -> set:
    """Crude proper-noun proxy (capitalized tokens, len>=3) -- not NER,
    deliberately cheap. Good enough to flag a mismatch, not precise enough
    to claim exact defect counts (see docs/data_summary.md's own caveat on
    this same heuristic)."""
    return set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", s))


def is_coherent(context: str, user_intent: str) -> bool:
    """A row is 'coherent' if every named entity in user_intent also shows
    up somewhere in context, or if user_intent has no named entity to check
    at all. False means the question likely wasn't generated with this
    specific context in view (docs/data_summary.md's MAlmasabi finding)."""
    names = entity_names(user_intent)
    if not names:
        return True
    ctx_lower = context.lower()
    return any(name.lower() in ctx_lower for name in names)


def proportional_stratified_sample(rows: List[dict], key_fn: Callable[[dict], object], n: int, seed: int = 0) -> List[dict]:
    """Sample exactly n rows total, splitting n across key_fn(row) groups
    proportionally to their share of the input -- so a fallback/backfill
    slice doesn't accidentally skew toward whichever group got shuffled to
    the front. The last group absorbs any rounding remainder so the total
    lands on exactly n rather than one-off from rounding every group down.

    Guardrails (ds-review MEDIUM finding): n must be non-negative, and the
    pool must actually contain at least n rows -- both fail loudly rather
    than silently returning a corrupted or short result. The caller
    (scripts/build_curated_pool.py) treats this function's output as
    authoritative for an exact 50/50 composition target; a silent shortfall
    or a negative-take corruption would only surface later as a subtly
    wrong split, not here where it's cheap to catch."""
    assert n >= 0, f"proportional_stratified_sample: n must be non-negative, got {n}"
    assert n <= len(rows), f"proportional_stratified_sample: requested n={n} exceeds pool size {len(rows)}"

    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)

    rng = random.Random(seed)
    total = len(rows)
    keys = list(groups.keys())
    result = []
    remaining_n = n
    for i, k in enumerate(keys):
        group = groups[k]
        is_last = i == len(keys) - 1
        take = remaining_n if is_last else min(round(n * len(group) / total), len(group), remaining_n)
        shuffled = group[:]
        rng.shuffle(shuffled)
        result.extend(shuffled[:take])
        remaining_n -= take

    assert len(result) == n, f"proportional_stratified_sample: produced {len(result)} rows, requested exactly {n}"
    return result
