"""
Inference wiring (Phase 1, docs/specs/IMPLEMENTATION_PLAN.md) -- turns a
trained model + tokenizer into the two call shapes the rest of the pipeline
needs: batched probability scores for eval (scripts/evaluate.py,
src/eval/report.py) and a single-string predict_fn for the red-team harness
(src/redteam/harness.py's PredictFn type, str -> Label).

Kept separate from src/model/train.py: training wiring and inference wiring
are different concerns that happen to share ModelConfig/encode_batch, not
one module's job.
"""

from typing import List

import torch

from src.data.schema import ContentSourceType, Example, Label
from src.model.train import ModelConfig, encode_batch


def predict_scores(model, tokenizer, examples: List[Example], config: ModelConfig) -> List[float]:
    """P(malicious) per example, via softmax over the two-class logits.
    LABEL_TO_ID (src/model/train.py) fixes MALICIOUS at index 1, so index 1
    of the softmax output is exactly the score this repo's eval metrics
    (src/eval/metrics.py, positive label = 1 = MALICIOUS) expect."""
    model.eval()
    batch = encode_batch(examples, tokenizer, config)
    labels = batch.pop("labels")  # forward pass here is inference-only, not training
    with torch.no_grad():
        logits = model(**batch).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[:, 1].tolist()


def predict_label_fn(model, tokenizer, config: ModelConfig, threshold: float = 0.5):
    """Returns a str -> Label closure matching
    src/redteam/harness.py's PredictFn -- the red-team runner script feeds
    seed content through this one string at a time, exactly as an agent
    would see a single tool output."""

    def predict_fn(content: str) -> Label:
        example = Example(
            example_id="inference-only",
            content_source_type=ContentSourceType.TOOL_OUTPUT,
            candidate_content=[content],
            label=Label.BENIGN,  # placeholder -- encode_batch needs a label field, unused for inference
        )
        score = predict_scores(model, tokenizer, [example], config)[0]
        return Label.MALICIOUS if score >= threshold else Label.BENIGN

    return predict_fn
