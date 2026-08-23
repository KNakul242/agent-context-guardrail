"""
Training loop (Phase 1, docs/specs/IMPLEMENTATION_PLAN.md). D13 (primary
model size) is resolved -- deberta-v3-small, ModelConfig's default. This
module has no __main__ of its own; scripts/train_primary.py is the real
entrypoint, and it enforces D24's hard gate (assert_self_authored_gate,
below) before calling run_training with real data. tests/model/test_train.py
exercises this against a handful of stub Examples and a tiny random HF model
(hf-internal-testing/tiny-random-DebertaV2Model) -- proof the plumbing works,
not a claim about model quality; that claim only gets made once a real run
against real data (gated on D24) actually happens.

backbone is a plain string field on ModelConfig specifically so that D13
(small/142M vs base/184M) and D7's ModernBERT fallback are both config edits,
not redesigns -- same reasoning D7's own gate already relies on.
"""

import random
from dataclasses import dataclass, field
from typing import Iterator, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.schema import Example, Label


def assert_self_authored_gate(self_authored_examples: List[Example]) -> None:
    """D24's hard gate, enforced as code rather than left as a convention to
    remember: a real training run may not begin with zero self-authored
    examples merged in, regardless of time remaining or how much of the
    curated public-source pool is ready. Called by scripts/train_primary.py
    before building the training set -- see docs/DECISIONS.md D24."""
    assert self_authored_examples, (
        "D24 hard gate: data/processed/self_authored.jsonl is missing or empty. "
        "A real training run may not begin with zero self-authored examples merged "
        "in -- see docs/DECISIONS.md D24. This does not flex regardless of time remaining."
    )

LABEL_TO_ID = {Label.BENIGN: 0, Label.MALICIOUS: 1}


@dataclass
class ModelConfig:
    backbone: str = "microsoft/deberta-v3-small"  # swap to -base for D13's other arm
    num_labels: int = 2
    max_length: int = 512


@dataclass
class TrainingRunConfig:
    """One real training run's parameters, plus its pre-registered success
    criterion -- DEVELOPMENT_RULES.md's TDD exception for training runs:
    you can't unit-test that a model will learn before running it, so the
    substitute discipline is writing down what "success" means before the
    run starts, then reporting the actual outcome against it honestly.
    success_criterion is required (non-empty) so this can't be skipped by
    just not filling it in."""

    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    success_criterion: str

    def __post_init__(self):
        assert self.success_criterion.strip(), "TrainingRunConfig requires a non-empty success_criterion"


@dataclass
class TrainingRunResult:
    loss_history: List[float] = field(default_factory=list)  # mean loss per epoch


def build_model_and_tokenizer(config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.backbone,
        num_labels=config.num_labels,
        # Pins the weight format explicitly -- DEVELOPMENT_RULES.md's DoD:
        # without this, huggingface_hub silently fetches both .bin and
        # .safetensors for the same checkpoint (observed: 742MB instead of
        # ~371MB during MPS verification).
        use_safetensors=True,
    )
    return model, tokenizer


def encode_batch(examples: List[Example], tokenizer, config: ModelConfig) -> dict:
    """window=1 only for now (schema.py's current default everywhere) --
    candidate_content[0] is the whole input. A window>1 example needs each
    entry encoded independently and pooled via src/model/aggregation.py's
    pool(), which happens on the *embeddings* the model produces, not here
    at the tokenization step -- wiring that in is real Phase-1-with-
    multi-turn-data work, not scaffolding, so it's not built until there's
    actual window>1 data to test it against."""
    texts = [ex.candidate_content[0] for ex in examples]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor([LABEL_TO_ID[ex.label] for ex in examples])
    return encoded


def train_step(model, batch: dict) -> torch.Tensor:
    """One forward pass, returns the loss. Caller runs .backward() and the
    optimizer step -- kept out of this function so it stays a pure,
    single-purpose wiring point tests/model/test_train.py can call directly."""
    output = model(**batch)
    return output.loss


def iterate_batches(examples: List[Example], batch_size: int, seed: int) -> Iterator[List[Example]]:
    """One epoch's worth of batches -- shuffles deterministically (fresh
    shuffle per call, so calling this again for the next epoch with the
    same seed does NOT repeat the same order; see run_training, which
    reseeds per-epoch off the run's base seed) and covers every example
    exactly once. Last batch may be smaller than batch_size."""
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    for i in range(0, len(shuffled), batch_size):
        yield shuffled[i : i + batch_size]


def run_training(model, tokenizer, examples: List[Example], model_config: ModelConfig, run_config: TrainingRunConfig) -> TrainingRunResult:
    """Real training loop. scripts/train_primary.py is the only caller that
    passes real data -- gated by assert_self_authored_gate() (D24) before it
    ever reaches this function. tests/model/test_train.py exercises this
    against 6 stub examples and a tiny random HF model to prove the loop
    (batching, forward, backward, optimizer step, per-epoch loss tracking)
    is wired correctly -- not to make any claim about a real model's
    quality."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config.learning_rate)
    result = TrainingRunResult()

    for epoch in range(run_config.epochs):
        epoch_losses = []
        # Reseeding per-epoch off (base seed, epoch) keeps the whole run
        # reproducible from run_config.seed alone while still reshuffling
        # each epoch -- a fixed single seed reused every epoch would train
        # on the identical batch order every time.
        for batch_examples in iterate_batches(examples, run_config.batch_size, seed=run_config.seed + epoch):
            batch = encode_batch(batch_examples, tokenizer, model_config)
            optimizer.zero_grad()
            loss = train_step(model, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        result.loss_history.append(sum(epoch_losses) / len(epoch_losses))

    return result
