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


def predict_scores(model, tokenizer, examples: List[Example], config: ModelConfig, batch_size: int = 32) -> List[float]:
    """P(malicious) per example, via softmax over the two-class logits.
    LABEL_TO_ID (src/model/train.py) fixes MALICIOUS at index 1, so index 1
    of the softmax output is exactly the score this repo's eval metrics
    (src/eval/metrics.py, positive label = 1 = MALICIOUS) expect.

    Chunks internally by batch_size rather than encoding every example in
    one shot (ds-review LOW finding, confirmed real: the full 1,293-row val
    split in one batch produced an actual MPS OOM during the first real eval
    run). Every caller -- scripts/evaluate.py, scripts/run_redteam.py via
    predict_label_fn -- gets the fix through this one shared choke point,
    same pattern as the device-placement and truncation-side fixes before it."""
    model.eval()
    scores: List[float] = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        batch = encode_batch(chunk, tokenizer, config)
        batch.pop("labels")  # forward pass here is inference-only, not training
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, 1].tolist())
    return scores


def predict_label_fn(model, tokenizer, config: ModelConfig, threshold: float = 0.5):
    """Returns a str -> Label closure matching
    src/redteam/harness.py's PredictFn -- the red-team runner script feeds
    seed content through this one string at a time, exactly as an agent
    would see a single tool output.

    Truncation note (ISSUE-9, docs/ISSUES.md): the Example built here
    carries no source/notes override, so encode_batch's _payload_is_appended
    check (src/model/train.py) never fires -- this always right-truncates,
    unlike training-time BIPIA insert_end rows, which are left-truncated.
    This is deliberate, not a bug to fix by copying that left-truncation
    here: D28's left-truncation was only possible because label-construction
    time has an oracle (BIPIA's own insert_end ground truth telling us
    exactly where the payload is). A real deployed guardrail has no such
    oracle at inference time -- it can't know in advance where in a long
    tool output a hidden payload might sit, so right-truncation (or, more
    accurately, whatever a real deployment's truncation policy is) may be
    MORE representative of production reality, not less. The actual
    consequence of this choice for long content-embedding red-team seeds
    (needle_in_haystack) is handled via
    src/redteam/harness.py's split_bypass_by_truncation_window() --
    reporting in-window vs. out-of-window bypass separately, not silently
    flipping this function's truncation behavior."""

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
