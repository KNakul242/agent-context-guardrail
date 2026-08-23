import pytest

from src.data.schema import ContentSourceType, Example, Label
from src.data.split import stratified_split


def _examples(n, source, label=Label.BENIGN):
    return [
        Example(
            example_id=f"{source}-{i}",
            content_source_type=ContentSourceType.TOOL_OUTPUT,
            candidate_content=["x"],
            label=label,
            source=source,
        )
        for i in range(n)
    ]


def test_split_preserves_total_count():
    examples = _examples(100, "a") + _examples(50, "b", Label.MALICIOUS)
    train, val, test = stratified_split(examples, seed=0)
    assert len(train) + len(val) + len(test) == 150


def test_every_example_appears_exactly_once():
    examples = _examples(100, "a") + _examples(50, "b", Label.MALICIOUS)
    train, val, test = stratified_split(examples, seed=0)
    all_ids = [ex.example_id for ex in train + val + test]
    assert sorted(all_ids) == sorted(ex.example_id for ex in examples)


def test_deterministic_given_same_seed():
    examples = _examples(100, "a") + _examples(50, "b", Label.MALICIOUS)
    r1 = stratified_split(examples, seed=42)
    r2 = stratified_split(examples, seed=42)
    assert [ex.example_id for ex in r1[0]] == [ex.example_id for ex in r2[0]]


def test_no_split_dominated_by_a_single_source():
    """The actual requirement this function exists to satisfy: every
    (source, label) stratum is represented in every split, roughly
    proportionally -- not concentrated into just one split."""
    examples = _examples(1000, "big_source") + _examples(50, "small_source", Label.MALICIOUS)
    train, val, test = stratified_split(examples, ratios=(0.8, 0.1, 0.1), seed=0)

    for split in (train, val, test):
        sources_present = {ex.source for ex in split}
        assert sources_present == {"big_source", "small_source"}


def test_split_ratios_approximately_respected_per_stratum():
    examples = _examples(1000, "a")
    train, val, test = stratified_split(examples, ratios=(0.8, 0.1, 0.1), seed=0)
    assert 750 <= len(train) <= 850
    assert 50 <= len(val) <= 150
    assert 50 <= len(test) <= 150


def test_tiny_stratum_still_gets_assigned_without_crashing():
    examples = _examples(3, "tiny")
    train, val, test = stratified_split(examples, seed=0)
    assert len(train) + len(val) + len(test) == 3


def test_ratios_must_sum_to_one():
    with pytest.raises(AssertionError):
        stratified_split(_examples(10, "a"), ratios=(0.5, 0.5, 0.5), seed=0)


def _bipia_document_family(doc_id, task="email", n_malicious=5):
    """One clean (benign) row plus several malicious rows all derived from
    the same underlying document -- the exact matched-pair shape D25 found:
    100% of curated BIPIA malicious rows share their clean_context_id with
    a benign row (docs/DECISIONS.md D25)."""
    benign = Example(
        example_id=f"bipia-{task}-clean-{doc_id}",
        content_source_type=ContentSourceType.TOOL_OUTPUT,
        candidate_content=["clean content"],
        label=Label.BENIGN,
        source="bipia",
        notes=f"task={task}; clean_context_id={doc_id}",
    )
    malicious = [
        Example(
            example_id=f"bipia-{task}-poisoned-{doc_id}-attack{i}",
            content_source_type=ContentSourceType.TOOL_OUTPUT,
            candidate_content=[f"poisoned content {i}"],
            label=Label.MALICIOUS,
            source="bipia",
            notes=f"task={task}; attack_category=cat; position=end; clean_context_id={doc_id}",
        )
        for i in range(n_malicious)
    ]
    return [benign] + malicious


def test_bipia_document_family_stays_in_one_split():
    """D25 (docs/DECISIONS.md): the same underlying document must not have
    its benign row in one split and a malicious derivative in another --
    that's the leakage D24 flagged and D25 confirmed (100% overlap)."""
    families = [_bipia_document_family(doc_id=i) for i in range(30)]
    examples = [ex for family in families for ex in family]

    train, val, test = stratified_split(examples, ratios=(0.8, 0.1, 0.1), seed=0)

    for split in (train, val, test):
        doc_ids_in_split = {ex.notes.split("clean_context_id=")[1] for ex in split}
        for doc_id in doc_ids_in_split:
            family_examples = [ex for ex in examples if ex.notes.endswith(f"clean_context_id={doc_id}")]
            assert all(ex in split for ex in family_examples), (
                f"document {doc_id}'s family is split across multiple splits"
            )


def test_bipia_family_split_includes_both_labels_together():
    families = [_bipia_document_family(doc_id=i) for i in range(10)]
    examples = [ex for family in families for ex in family]

    train, val, test = stratified_split(examples, ratios=(0.8, 0.1, 0.1), seed=0)

    for split in (train, val, test):
        if not split:
            continue
        labels_present = {ex.label for ex in split}
        # Every split that has any BIPIA content at all should have both
        # labels, since every family carries both -- not one split
        # accidentally ending up benign-only or malicious-only.
        assert labels_present == {Label.BENIGN, Label.MALICIOUS}


def test_non_bipia_sources_still_split_per_example_not_grouped():
    """Grouping is specific to BIPIA's matched-pair structure -- other
    sources have no such document-family relationship, so each row should
    still be its own independent stratification unit, exactly as before."""
    examples = _examples(100, "prodnull") + _examples(50, "prodnull", Label.MALICIOUS)
    train, val, test = stratified_split(examples, ratios=(0.8, 0.1, 0.1), seed=0)
    # 80% of 100 benign (singleton groups) + 80% of 50 malicious = 120, not
    # ~80 -- each row is its own group here, so per-stratum rounding lands
    # on the exact 80/10/10 split of each label count independently.
    assert len(train) == 120
