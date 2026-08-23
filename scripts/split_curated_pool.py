#!/usr/bin/env python3
"""
Materializes data/processed/{train,val,test}.jsonl from
data/processed/curated_pool.jsonl using src/data/split.py's document-
group-aware stratified_split() (docs/DECISIONS.md D25 -- BIPIA rows are
grouped by underlying document identity so a document's benign form and
its malicious derivatives always land in the same split).

Run scripts/build_curated_pool.py first if curated_pool.jsonl doesn't
exist yet. Does not include data/processed/self_authored.jsonl -- per
docs/DECISIONS.md D24, that slice merges in separately later; re-run this
script after it does.

Usage:
    python3 scripts/split_curated_pool.py
"""

import dataclasses
import json
from pathlib import Path

from src.data.schema import ContentSourceType, Example, InjectionTechnique, Label, validate
from src.data.split import document_group_key, stratified_split

IN_PATH = Path("data/processed/curated_pool.jsonl")
RATIOS = (0.8, 0.1, 0.1)
SEED = 0


def _load(path: Path) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            examples.append(
                Example(
                    example_id=d["example_id"],
                    content_source_type=ContentSourceType(d["content_source_type"]),
                    candidate_content=d["candidate_content"],
                    label=Label(d["label"]),
                    agent_task_context=d.get("agent_task_context"),
                    technique=InjectionTechnique(d["technique"]) if d.get("technique") else None,
                    source=d["source"],
                    is_redteam=d.get("is_redteam", False),
                    notes=d.get("notes", ""),
                )
            )
    return examples


def _write(path: Path, examples: list) -> None:
    with open(path, "w") as f:
        for ex in examples:
            d = dataclasses.asdict(ex)
            d["content_source_type"] = ex.content_source_type.value
            d["label"] = ex.label.value
            d["technique"] = ex.technique.value if ex.technique else None
            f.write(json.dumps(d) + "\n")


def main():
    examples = _load(IN_PATH)
    for ex in examples:
        validate(ex)

    train, val, test = stratified_split(examples, ratios=RATIOS, seed=SEED)

    # Verify D25's guarantee held, not just assume the fix still works.
    group_to_split = {}
    leaks = 0
    for name, split in [("train", train), ("val", val), ("test", test)]:
        for ex in split:
            if ex.source != "bipia":
                continue
            key = document_group_key(ex)
            if key in group_to_split and group_to_split[key] != name:
                leaks += 1
            group_to_split[key] = name
    assert leaks == 0, f"{leaks} BIPIA document groups leaked across splits -- do not write output"

    for name, split in [("train", train), ("val", val), ("test", test)]:
        _write(Path(f"data/processed/{name}.jsonl"), split)
        print(f"{name}: {len(split)} rows -> data/processed/{name}.jsonl")

    print(f"\nBIPIA document groups: {len(group_to_split)}, leaks: {leaks}")
    print(f"total: {len(train) + len(val) + len(test)} (source: {len(examples)})")


if __name__ == "__main__":
    main()
