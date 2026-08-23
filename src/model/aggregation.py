"""
Multi-turn aggregation module (D8a, docs/DECISIONS.md).

D8a's design is embed-then-pool: encode each tool output in the
`candidate_content` window independently (that part happens in the model's
forward pass, not here), then pool the resulting fixed-size embeddings into
one vector with the function below. Deliberately NOT concatenate-then-encode
-- concatenating raw text before encoding would redistribute the same
512-token budget across more content instead of extending it, which is the
opposite of what a longer window is supposed to buy us.

This module only implements the pooling step and is intentionally decoupled
from the actual encoder (no model dependency here) -- Phase 1 is what wires
a trained encoder's per-output embeddings through this function. That
decoupling is also why this is fully unit-testable now, without a trained
model: pooling is pure tensor math, testable with hand-built vectors.
"""

from typing import List

import torch

_STRATEGIES = {
    "mean": lambda stacked: stacked.mean(dim=0),
    "max": lambda stacked: stacked.max(dim=0).values,
}


def pool(embeddings: List[torch.Tensor], strategy: str = "mean") -> torch.Tensor:
    """Pool a window of independently-encoded tool-output embeddings into
    one fixed-size vector.

    A window of size 1 (the current default everywhere in this repo, per
    schema.py) must come back out unchanged regardless of strategy -- that
    equivalence to the stateless single-input baseline is the whole point
    of D8a's window design, not an incidental property, so every strategy
    added here has to preserve it.
    """
    assert len(embeddings) >= 1, "pool() requires at least one embedding (mirrors schema.py's window invariant)"

    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown pooling strategy {strategy!r}; choose one of {sorted(_STRATEGIES)}")

    stacked = torch.stack(embeddings)
    return _STRATEGIES[strategy](stacked)
