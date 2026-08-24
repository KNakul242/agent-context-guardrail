"""
JSONL read/write for Example (docs/specs/DEVELOPMENT_RULES.md TDD scope).

Factored out of scripts/build_curated_pool.py and scripts/split_curated_pool.py,
which each hand-rolled the same load/write logic (Phase 0) -- Phase 1's
train/eval/red-team entrypoints need the identical round trip a third and
fourth time, which is the point past which copy-pasting it again stops being
the simpler option.
"""

import dataclasses
import json
from pathlib import Path
from typing import List, Union

from src.data.schema import ContentSourceType, Example, InjectionTechnique, Label


def load_examples_jsonl(path: Union[str, Path]) -> List[Example]:
    """Missing file -> empty list, not an error -- callers (e.g. the D24
    self-authored gate check) need "not yet produced" and "produced but
    empty" to both read as zero rows without a try/except at every call site."""
    path = Path(path)
    if not path.exists():
        return []
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
                    source=d.get("source", "self_authored"),
                    is_redteam=d.get("is_redteam", False),
                    notes=d.get("notes", ""),
                )
            )
    return examples


def write_examples_jsonl(path: Union[str, Path], examples: List[Example]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            d = dataclasses.asdict(ex)
            d["content_source_type"] = ex.content_source_type.value
            d["label"] = ex.label.value
            d["technique"] = ex.technique.value if ex.technique else None
            f.write(json.dumps(d) + "\n")
