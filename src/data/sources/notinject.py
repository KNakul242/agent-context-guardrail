"""
leolee99/NotInject -> Example schema mapping (docs/DATA_SOURCES.md).

MIT, no gate, 339 rows total across three splits (NotInject_one/two/three,
113 rows each) keyed by trigger-word count in `word_list`. Every row is a
hard-negative benign prompt: it contains an imperative/trigger word (e.g.
"ignore") without being an actual injection attempt (D19 — supplement to
an existing balanced pool, not a primary source needing its own malicious
half; InjecGuard paper Section 4.1 is the methodology this construction is
based on).
"""

import json
from pathlib import Path
from typing import Iterator

from src.data.schema import ContentSourceType, Example, Label

RAW_DIR = Path("data/raw/notinject")
SPLITS = ["NotInject_one", "NotInject_two", "NotInject_three"]


def pull_raw(out_dir: Path = RAW_DIR) -> None:
    """Download the dataset untouched and write each split to JSONL under out_dir."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("leolee99/NotInject")
    for split in SPLITS:
        with open(out_dir / f"{split}.jsonl", "w") as f:
            for row in ds[split]:
                f.write(json.dumps(row) + "\n")


def map_row_to_example(row: dict, split: str, index: int) -> Example:
    return Example(
        example_id=f"notinject-{split}-{index}",
        content_source_type=ContentSourceType.TOOL_OUTPUT,
        candidate_content=[row["prompt"]],
        label=Label.BENIGN,
        technique=None,
        source="notinject",
        is_redteam=False,
        notes=f"split={split}; category={row['category']}; trigger_words={row['word_list']}",
    )


def load_examples(raw_dir: Path = RAW_DIR) -> Iterator[Example]:
    """Read the raw JSONL pulled by pull_raw() and map every row to an Example."""
    for split in SPLITS:
        path = raw_dir / f"{split}.jsonl"
        with open(path) as f:
            for index, line in enumerate(f):
                row = json.loads(line)
                yield map_row_to_example(row, split=split, index=index)
