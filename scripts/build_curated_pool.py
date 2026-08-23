#!/usr/bin/env python3
"""
Regenerates data/processed/curated_pool.jsonl from the raw pulls.

Orchestrates the composition pass described in docs/data_summary.md §0 and
docs/DECISIONS.md D13/D25 -- this script didn't exist when that pass was
first run (it was executed as inline scripts in an interactive session),
which was itself a gap flagged against DEVELOPMENT_RULES.md's "run is
reproducible... exact run command documented" DoD item. This closes it:
every parameter below (cap=6, seed=0, the 50/50 target) is exactly what
produced the numbers already reported in data_summary.md, not a fresh
re-derivation.

Usage:
    python3 scripts/build_curated_pool.py [--pull]

    --pull    Also (re-)run each source's pull_raw() first. Off by default
              since prodnull/MAlmasabi need HF auth (`hf auth login`) and a
              full BIPIA re-materialization is ~74MB / a couple minutes --
              skip it if data/raw/ is already populated from a prior run.

Does not touch data/processed/self_authored.jsonl -- per docs/DECISIONS.md
D24, that slice is merged in separately, opportunistically, not as part of
this script.
"""

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

from src.data.curate import (
    classify_domain,
    is_coherent,
    proportional_stratified_sample,
    stratified_cap,
)
from src.data.schema import ContentSourceType, Label, validate
from src.data.sources import malmasabi, notinject, prodnull
from src.data.sources.bipia import (
    RAW_DIR as BIPIA_RAW_DIR,
    TASKS as BIPIA_TASKS,
    full_benign_corpus,
    map_pair_to_examples,
)
from src.data.sources.bipia import pull_raw as bipia_pull_raw
from src.data.sources.malmasabi import RAW_DIR as MALMASABI_RAW_DIR
from src.data.sources.malmasabi import pull_raw as malmasabi_pull_raw
from src.data.sources.notinject import RAW_DIR as NOTINJECT_RAW_DIR
from src.data.sources.notinject import pull_raw as notinject_pull_raw
from src.data.sources.prodnull import RAW_DIR as PRODNULL_RAW_DIR
from src.data.sources.prodnull import pull_raw as prodnull_pull_raw

BIPIA_MALICIOUS_CAP = 6  # docs/DECISIONS.md: 15 (the originally suggested cap)
# would have produced 9,000 rows, over double the 3,500-4,000 target; 6
# lands at exactly 3,600.
SEED = 0
OUT_PATH = Path("data/processed/curated_pool.jsonl")


def _maybe_pull(name: str, raw_dir: Path, pull_fn, force: bool) -> None:
    if not force and raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"[{name}] raw data already present at {raw_dir}, skipping pull (use --pull to force)")
        return
    print(f"[{name}] pulling raw data...")
    pull_fn()


def build(force_pull: bool) -> list:
    _maybe_pull("notinject", NOTINJECT_RAW_DIR, notinject_pull_raw, force_pull)
    _maybe_pull("prodnull", PRODNULL_RAW_DIR, prodnull_pull_raw, force_pull)
    _maybe_pull("malmasabi", MALMASABI_RAW_DIR, malmasabi_pull_raw, force_pull)
    _maybe_pull("bipia", BIPIA_RAW_DIR, bipia_pull_raw, force_pull)

    # --- BIPIA malicious: stratified sample, not full combinatorial volume ---
    all_poisoned = []
    for task in BIPIA_TASKS:
        with open(BIPIA_RAW_DIR / f"{task}_poisoned.jsonl") as f:
            for line in f:
                row = json.loads(line)
                row["_task"] = task
                all_poisoned.append(row)

    def key_fn(r):
        return (r["_task"], r["attack_category"], r["attack_index"], r["position"])

    bipia_mal_sample = stratified_cap(all_poisoned, key_fn=key_fn, cap=BIPIA_MALICIOUS_CAP, seed=SEED)
    bipia_malicious_examples = [
        map_pair_to_examples(row, task=row["_task"], kind="poisoned", row_index=i)
        for i, row in enumerate(bipia_mal_sample)
    ]
    print(f"BIPIA malicious: {len(bipia_malicious_examples)} (cap={BIPIA_MALICIOUS_CAP})")

    # --- prodnull, NotInject: unchanged, full volume ---
    prodnull_examples = list(prodnull.load_examples())
    notinject_examples = list(notinject.load_examples())

    # --- BIPIA benign: corrected full corpus (D25) ---
    benign_by_task = full_benign_corpus(seed=SEED)
    bipia_benign_examples = []
    for task, rows in benign_by_task.items():
        bipia_benign_examples.extend(
            map_pair_to_examples(row, task=task, kind="clean", row_index=i) for i, row in enumerate(rows)
        )
    print(f"BIPIA benign: {len(bipia_benign_examples)}")

    # --- MAlmasabi: coherence-filtered, domain-stratified fallback to close the gap ---
    total_malicious = len(bipia_malicious_examples) + sum(1 for e in prodnull_examples if e.label == Label.MALICIOUS)
    baseline_benign = len(notinject_examples) + sum(1 for e in prodnull_examples if e.label == Label.BENIGN)
    running = baseline_benign + len(bipia_benign_examples)
    gap = total_malicious - running
    print(f"target malicious: {total_malicious}; benign before MAlmasabi: {running}; gap: {gap}")

    with open(MALMASABI_RAW_DIR / "train.jsonl") as f:
        rows = [json.loads(l) for l in f]
    benign_pool = [r for r in rows if r["label"] == 0]
    coherent = [r for r in benign_pool if is_coherent(r["context"], r["user_intent"])]
    for r in coherent:
        r["_domain"] = classify_domain(r["context"])
    backfill = proportional_stratified_sample(coherent, key_fn=lambda r: r["_domain"], n=gap, seed=SEED)
    malmasabi_examples = [malmasabi.map_row_to_example(row, index=i) for i, row in enumerate(backfill)]
    print(f"MAlmasabi backfill: {len(malmasabi_examples)}")

    # --- Combine, validate ---
    curated = notinject_examples + prodnull_examples + bipia_malicious_examples + bipia_benign_examples + malmasabi_examples

    non_tool_output = [ex for ex in curated if ex.content_source_type != ContentSourceType.TOOL_OUTPUT]
    assert not non_tool_output, f"{len(non_tool_output)} rows are not content_source_type=TOOL_OUTPUT"
    for ex in curated:
        validate(ex)

    return curated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", action="store_true", help="Also re-run each source's pull_raw() first")
    args = parser.parse_args()

    curated = build(force_pull=args.pull)

    by_label = Counter(ex.label.value for ex in curated)
    by_source_label_technique = Counter(
        (ex.source, ex.label.value, ex.technique.value if ex.technique else "n/a") for ex in curated
    )

    print(f"\nTOTAL: {len(curated)}  ({dict(by_label)})")
    for k in sorted(by_source_label_technique):
        print(" ", k, by_source_label_technique[k])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for ex in curated:
            d = dataclasses.asdict(ex)
            d["content_source_type"] = ex.content_source_type.value
            d["label"] = ex.label.value
            d["technique"] = ex.technique.value if ex.technique else None
            f.write(json.dumps(d) + "\n")

    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
