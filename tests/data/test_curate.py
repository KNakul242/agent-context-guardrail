from src.data.curate import (
    classify_domain,
    entity_names,
    is_coherent,
    proportional_stratified_sample,
    stratified_cap,
)


# --- stratified_cap ---

def test_stratified_cap_respects_cap_per_group():
    rows = [{"g": "a", "i": i} for i in range(20)] + [{"g": "b", "i": i} for i in range(3)]
    result = stratified_cap(rows, key_fn=lambda r: r["g"], cap=5, seed=0)
    assert sum(1 for r in result if r["g"] == "a") == 5
    assert sum(1 for r in result if r["g"] == "b") == 3  # group smaller than cap: keep all


def test_stratified_cap_deterministic():
    rows = [{"g": "a", "i": i} for i in range(20)]
    r1 = stratified_cap(rows, key_fn=lambda r: r["g"], cap=5, seed=7)
    r2 = stratified_cap(rows, key_fn=lambda r: r["g"], cap=5, seed=7)
    assert r1 == r2


# --- classify_domain ---

def test_classify_domain_table():
    ctx = "| Year | Winner |\n|---|---|\n| 2005 | Galaxy |"
    assert classify_domain(ctx) == "table"


def test_classify_domain_email():
    ctx = "SUBJECT: Invoice due\nEMAIL_FROM: billing@example.com\nHi there..."
    assert classify_domain(ctx) == "email"


def test_classify_domain_code():
    ctx = "```\nimport requests\ndef test():\n    pass\n```\nTraceback (most recent call last):"
    assert classify_domain(ctx) == "code"


def test_classify_domain_unclassified_for_plain_prose():
    ctx = "A news article about a court case involving a local business owner."
    assert classify_domain(ctx) == "unclassified"


# --- entity_names / is_coherent ---

def test_entity_names_extracts_capitalized_tokens():
    assert entity_names("Find the value paid to Ganesha?") == {"Find", "Ganesha"}


def test_is_coherent_true_when_entity_present_in_context():
    assert is_coherent(context="A payment to Ganesha was recorded.", user_intent="Find the $ paid to Ganesha")


def test_is_coherent_false_when_entity_absent_from_context():
    assert not is_coherent(
        context="<?php $conn = new mysqli(...); ?>",
        user_intent="Find the $ value paid to Ganesha?",
    )


def test_is_coherent_true_when_no_entity_to_check():
    assert is_coherent(context="anything at all", user_intent="summarize this")


# --- proportional_stratified_sample ---

def test_proportional_sample_hits_requested_total():
    rows = [{"d": "table"}] * 700 + [{"d": "email"}] * 200 + [{"d": "code"}] * 100
    result = proportional_stratified_sample(rows, key_fn=lambda r: r["d"], n=100, seed=0)
    assert len(result) == 100


def test_proportional_sample_preserves_rough_proportions():
    rows = [{"d": "table"}] * 700 + [{"d": "email"}] * 200 + [{"d": "code"}] * 100
    result = proportional_stratified_sample(rows, key_fn=lambda r: r["d"], n=100, seed=0)
    counts = {"table": 0, "email": 0, "code": 0}
    for r in result:
        counts[r["d"]] += 1
    assert 60 <= counts["table"] <= 80  # ~70
    assert 10 <= counts["email"] <= 30  # ~20
    assert 0 <= counts["code"] <= 20  # ~10
