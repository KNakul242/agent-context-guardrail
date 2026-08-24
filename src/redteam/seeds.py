"""
Loader for the manual red-team seed corpus (docs/specs/IMPLEMENTATION_PLAN.md
Phase 1, D10's manual-seed layer).

data/redteam/seeds.jsonl is authored content, not derived/pulled data -- each
seed is a hand-constructed IPI attempt covering one InjectionTechnique
category (D18/D22's method-based taxonomy), framed as a plausible tool-output
carrier document (repo file, email, table cell, API response, webpage
excerpt) matching this project's actual carrier-document coverage (D23).

This is red-team probe material, not training data -- it is never merged
into curated_pool.jsonl directly. Only confirmed bypasses
(src/redteam/harness.py's harvest_bypasses()) become training Examples, per
the harvest-retrain step in D10/IMPLEMENTATION_PLAN.md Phase 1. It therefore
does not interact with D24's self-authored-training-data gate at all.

PAYLOAD_SPLIT note: D8a names payload-splitting-across-turns as a scope
limitation by construction under window=1 (no second turn to split across).
The PAYLOAD_SPLIT seeds here instead split the payload across *fields within
one document* (e.g. a key/value table where no single field reads as a
complete instruction) -- a real, in-scope structural-evasion test distinct
from the acknowledged cross-turn gap, not a workaround for it.

LOW_RESOURCE_LANGUAGE note: per D22, these are machine-approximate
translations, not verified by a native/fluent speaker -- treat any
LOW_RESOURCE_LANGUAGE-segmented bypass-rate result in the write-up as
approximate for the same reason D22 already flags BIPIA's own bucket.
"""

import json
from pathlib import Path
from typing import List, Union

from src.redteam.harness import RedTeamSeed
from src.data.schema import InjectionTechnique

DEFAULT_SEEDS_PATH = Path("data/redteam/seeds.jsonl")


def load_seeds(path: Union[str, Path] = DEFAULT_SEEDS_PATH) -> List[RedTeamSeed]:
    path = Path(path)
    if not path.exists():
        return []
    seeds = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            seeds.append(
                RedTeamSeed(
                    seed_id=d["seed_id"],
                    technique=InjectionTechnique(d["technique"]),
                    content=d["content"],
                    injection_marker=d.get("injection_marker"),
                )
            )
    return seeds
