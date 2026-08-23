"""
Training loop (Phase 1, docs/specs/IMPLEMENTATION_PLAN.md). D13 (primary
model size) is resolved -- deberta-v3-small, ModelConfig's default. This
module has no __main__ of its own; scripts/train_primary.py is the real
entrypoint. tests/model/test_train.py exercises this against a handful of
stub Examples and a tiny random HF model
(hf-internal-testing/tiny-random-DebertaV2Model) -- proof the plumbing works,
not a claim about model quality; that claim only gets made once a real run
against real data actually happens.

backbone is a plain string field on ModelConfig specifically so that D13
(small/142M vs base/184M) and D7's ModernBERT fallback are both config edits,
not redesigns -- same reasoning D7's own gate already relies on.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.schema import Example, Label


def assert_self_authored_gate(self_authored_examples: List[Example]) -> None:
    """D24's hard block on zero self-authored examples was explicitly
    lifted (D27) -- a real run may now proceed with none merged in. This
    function is kept, not deleted, so every run's self-authored count is
    still visible in output rather than silently omitted; it no longer
    raises. Called by scripts/train_primary.py before building the
    training set."""
    print(f"self-authored examples merged into this run: {len(self_authored_examples)} (D27: not gating)")


LABEL_TO_ID = {Label.BENIGN: 0, Label.MALICIOUS: 1}


@dataclass
class ModelConfig:
    backbone: str = "microsoft/deberta-v3-small"  # swap to -base for D13's other arm
    num_labels: int = 2
    max_length: int = 512
    device: Optional[str] = None  # None = auto-detect via resolve_device (mps > cuda > cpu)


def resolve_device(config: ModelConfig, mps_available: Optional[bool] = None, cuda_available: Optional[bool] = None) -> str:
    """Explicit config.device always wins. Otherwise auto-detect, preferring
    MPS (D7's smoke test already verified DeBERTa-v3-base trains cleanly on
    the local M-series MPS backend -- this is what makes the "trains in
    minutes on MPS" claim in CLAUDE.md actually true instead of aspirational).
    mps_available/cuda_available are injectable so this stays a pure,
    hardware-independent function for tests -- production callers never pass
    them and get torch's real availability checks."""
    if config.device is not None:
        return config.device
    if mps_available is None:
        mps_available = torch.backends.mps.is_available()
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if mps_available:
        return "mps"
    if cuda_available:
        return "cuda"
    return "cpu"


DEFAULT_FULL_RUN_CHECKPOINT_DIR = Path("models/primary")
DEFAULT_SMOKE_TEST_CHECKPOINT_DIR = Path("models/smoke_test")


def epoch_checkpoint_subdir(base_dir: Path, epoch: int) -> Path:
    """1-indexed (epoch=0 -> epoch_1) to match manifest.json's human-readable
    "completed_epochs" field. Each epoch gets its own directory rather than
    all epochs overwriting one shared path -- save_pretrained() writes
    several files (config.json, model.safetensors, tokenizer files) and is
    not atomic as a whole; an interrupted write to a shared path could
    corrupt the directory while destroying the last known-good epoch's
    state, with nothing left to recover (docs/ISSUES.md ISSUE-6 correction,
    caught by peer review after the original overwrite-in-place design)."""
    return base_dir / f"epoch_{epoch + 1}"


def resolve_checkpoint_dir(output_dir_arg: Optional[str], limit: Optional[int], run_id: Optional[str] = None) -> Path:
    """An explicit output_dir_arg always wins (run_id is ignored in that
    case -- an explicit dir is a deliberate, specific choice). Otherwise: a
    --limit run (a smoke test, not a real training pass) defaults to
    DEFAULT_SMOKE_TEST_CHECKPOINT_DIR, never DEFAULT_FULL_RUN_CHECKPOINT_DIR
    -- both used to default to the same hardcoded path, and a --limit
    diagnostic silently overwrote a full run's checkpoint mid-session
    (docs/ISSUES.md ISSUE-3, found via peer review).

    run_id, when given for a --limit run, nests under the smoke-test
    default so two different diagnostic runs don't collide with each other
    the same way -- a residual gap in the ISSUE-3 fix, also caught by peer
    review (docs/ISSUES.md ISSUE-5). Meaningless for a full run (exactly
    one models/primary/ by design), so ignored there."""
    if output_dir_arg is not None:
        return Path(output_dir_arg)
    if limit is None:
        return DEFAULT_FULL_RUN_CHECKPOINT_DIR
    return DEFAULT_SMOKE_TEST_CHECKPOINT_DIR / run_id if run_id is not None else DEFAULT_SMOKE_TEST_CHECKPOINT_DIR


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
    # ISSUE-1 (docs/ISSUES.md): AdamW's default eps=1e-8 underflows in its
    # sqrt/addcdiv update math on PyTorch's MPS backend, corrupting model
    # weights to NaN/Inf on the very first optimizer.step() -- confirmed by
    # direct isolation (CPU clean, MPS+SGD clean, MPS+AdamW eps=1e-8 NaNs
    # immediately, MPS+AdamW eps=1e-6 stays clean). 1e-6 is the standard
    # documented mitigation for this class of MPS numerical bug. Left
    # tunable rather than hardcoded since this is a real, hardware-specific
    # hyperparameter, not an implementation detail.
    adam_eps: float = 1e-6

    def __post_init__(self):
        assert self.success_criterion.strip(), "TrainingRunConfig requires a non-empty success_criterion"


@dataclass
class TrainingRunResult:
    loss_history: List[float] = field(default_factory=list)  # mean loss per epoch


def build_model_and_tokenizer(config: ModelConfig, seed: Optional[int] = None):
    """seed, if given, is applied via torch.manual_seed() (+
    torch.cuda.manual_seed_all() when CUDA is present) BEFORE from_pretrained
    constructs the model -- this is the only point in the pipeline where it
    matters. The classifier/pooler head's random init happens inside
    from_pretrained itself; TrainingRunConfig.seed (used elsewhere, in
    iterate_batches) only ever controlled batch-shuffle order via a separate
    random.Random(seed) instance, never torch's own global RNG. Without
    this, DEVELOPMENT_RULES.md's DoD ("Run is reproducible: seed fixed")
    wasn't actually satisfied for the one thing most consequential to
    reproduce -- confirmed by peer review during this session, the model
    came within a hair of total collapse (docs/ISSUES.md ISSUE-3) on an
    init nobody could have reproduced to debug."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.backbone,
        num_labels=config.num_labels,
        # Pins the weight format explicitly -- DEVELOPMENT_RULES.md's DoD:
        # without this, huggingface_hub silently fetches both .bin and
        # .safetensors for the same checkpoint (observed: 742MB instead of
        # ~371MB during MPS verification).
        use_safetensors=True,
        # ISSUE-4 (docs/ISSUES.md): without this, transformers>=4.44 loads
        # whatever dtype the hub checkpoint was stored in --
        # microsoft/deberta-v3-small is stored in float16, and training a
        # freshly-initialized classifier head in fp16 with no GradScaler/
        # loss-scaling anywhere in run_training silently underflows its
        # gradients to near-zero. Same "pin the numerically consequential
        # format explicitly" discipline as use_safetensors above.
        dtype=torch.float32,
    )
    model = model.to(resolve_device(config))
    return model, tokenizer


def _payload_is_appended(ex: Example) -> bool:
    """True for BIPIA insert_end rows (third_party/BIPIA/bipia/data/utils.py:
    insert_end returns context + "\\n" + attack -- the injected instruction is
    literally the last tokens of the document). notes carries position=end
    exactly as src/data/sources/bipia.py's map_pair_to_examples writes it.
    These rows need left- rather than right-truncation (see encode_batch) --
    verified against real data (ds-review HIGH finding): 89/1200 (7.4%) of
    curated BIPIA insert_end malicious rows exceed 512 tokens pre-truncation,
    which under HF's default right-truncation silently strips the appended
    attack string while the row still carries label=MALICIOUS."""
    return ex.source == "bipia" and "position=end" in ex.notes


def encode_batch(examples: List[Example], tokenizer, config: ModelConfig) -> dict:
    """window=1 only for now (schema.py's current default everywhere) --
    candidate_content[0] is the whole input. A window>1 example needs each
    entry encoded independently and pooled via src/model/aggregation.py's
    pool(), which happens on the *embeddings* the model produces, not here
    at the tokenization step -- wiring that in is real Phase-1-with-
    multi-turn-data work, not scaffolding, so it's not built until there's
    actual window>1 data to test it against.

    Truncation side is per-row, not a single tokenizer-wide setting: BIPIA
    insert_end rows (_payload_is_appended) are left-truncated so the appended
    payload survives; every other row keeps HF's ordinary right-truncation
    default. Tokenizes each side's subset separately (tokenizer.truncation_side
    is mutable global state on the tokenizer instance, restored in a finally
    block so a later unrelated call -- e.g. predict_scores on the next batch --
    never inherits whatever side this call last left it in), then merges both
    subsets back into original row order via tokenizer.pad(), the standard HF
    pattern for padding already-tokenized (padding=False) sequences."""
    texts = [ex.candidate_content[0] for ex in examples]
    left_indices = [i for i, ex in enumerate(examples) if _payload_is_appended(ex)]
    right_indices = [i for i in range(len(examples)) if i not in set(left_indices)]

    features = [None] * len(examples)
    original_truncation_side = tokenizer.truncation_side
    try:
        for side, indices in (("left", left_indices), ("right", right_indices)):
            if not indices:
                continue
            tokenizer.truncation_side = side
            encoded = tokenizer(
                [texts[i] for i in indices],
                truncation=True,
                max_length=config.max_length,
                padding=False,
            )
            for local_i, orig_i in enumerate(indices):
                features[orig_i] = {k: encoded[k][local_i] for k in encoded.keys()}
    finally:
        tokenizer.truncation_side = original_truncation_side

    padded = tokenizer.pad(features, padding=True, return_tensors="pt")
    padded["labels"] = torch.tensor([LABEL_TO_ID[ex.label] for ex in examples])
    return padded


def train_step(model, batch: dict) -> torch.Tensor:
    """One forward pass, returns the loss. Caller runs .backward() and the
    optimizer step -- kept out of this function so it stays a pure,
    single-purpose wiring point tests/model/test_train.py can call directly.
    Moves the batch to the model's own device (set by build_model_and_tokenizer's
    resolve_device() call) rather than taking a device param -- the model is
    always the source of truth for where a batch needs to live."""
    batch = {k: v.to(model.device) for k, v in batch.items()}
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


def run_training(
    model,
    tokenizer,
    examples: List[Example],
    model_config: ModelConfig,
    run_config: TrainingRunConfig,
    on_epoch_end: Optional[Callable[[int, float], None]] = None,
) -> TrainingRunResult:
    """Real training loop. scripts/train_primary.py is the only caller that
    passes real data; assert_self_authored_gate() runs first but only logs
    the self-authored count now (D27), it does not gate. tests/model/test_train.py
    exercises this against 6 stub examples and a tiny random HF model to prove
    the loop (batching, forward, backward, optimizer step, per-epoch loss
    tracking) is wired correctly -- not to make any claim about a real
    model's quality.

    on_epoch_end(epoch_index, mean_loss), if given, is called after every
    epoch completes (ISSUE-5, docs/ISSUES.md) -- this is the hook
    scripts/train_primary.py uses to checkpoint after every epoch instead
    of only once after the whole run returns, so a late-stage failure on a
    long run doesn't lose everything. Kept as an injectable callback rather
    than calling save_pretrained directly in here, so this function stays
    file-I/O-free and testable with a stub, same split as every other
    real-vs-stub boundary in this module."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config.learning_rate, eps=run_config.adam_eps)
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
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        result.loss_history.append(mean_loss)
        if on_epoch_end is not None:
            on_epoch_end(epoch, mean_loss)

    return result
