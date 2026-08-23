"""
prodnull/prompt-injection-repo-dataset -> Example schema mapping
(docs/DATA_SOURCES.md, D19).

Apache 2.0, gated (HF account + terms). Primary bulk source: coding-agent /
repo-file specific content, near-exact match to our surface. Flat
{text, label} schema, single "train" split, label 1 == malicious (confirmed
by count: 2,916 rows at label==1 matches the dataset card's malicious count
exactly).

Known gap, not silently papered over: the dataset card documents a 24-way
attack-category taxonomy, but that taxonomy isn't a queryable column in this
release -- only text/label. Re-deriving 24 categories from raw text would be
its own small classification project (keyword heuristics would be guessing,
not deriving), which is out of scope for a data-pull script. Every malicious
row here gets InjectionTechnique.OTHER rather than a fabricated guess -- see
docs/DATA_SOURCES.md for this flagged as a follow-up, since it means
prodnull's malicious examples won't contribute to D10's per-category
segmentation until that's revisited.
"""

import json
from pathlib import Path
from typing import Iterator

from src.data.schema import ContentSourceType, Example, InjectionTechnique, Label

RAW_DIR = Path("data/raw/prodnull")


def pull_raw(out_dir: Path = RAW_DIR) -> None:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("prodnull/prompt-injection-repo-dataset")["train"]
    with open(out_dir / "train.jsonl", "w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")


def map_row_to_example(row: dict, index: int) -> Example:
    is_malicious = row["label"] == 1
    return Example(
        example_id=f"prodnull-{index}",
        content_source_type=ContentSourceType.TOOL_OUTPUT,
        candidate_content=[row["text"]],
        label=Label.MALICIOUS if is_malicious else Label.BENIGN,
        technique=InjectionTechnique.OTHER if is_malicious else None,
        source="prodnull",
        is_redteam=False,
        notes="category not available in this HF release; see module docstring" if is_malicious else "",
    )


def load_examples(raw_dir: Path = RAW_DIR) -> Iterator[Example]:
    path = raw_dir / "train.jsonl"
    with open(path) as f:
        for index, line in enumerate(f):
            yield map_row_to_example(json.loads(line), index=index)
