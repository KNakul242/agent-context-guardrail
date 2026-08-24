#!/usr/bin/env python3
"""
Real training entrypoint for the primary detector (Phase 1,
docs/specs/IMPLEMENTATION_PLAN.md; D13 backbone resolved).

Loads data/processed/train.jsonl (the curated public-source pool) plus
data/processed/self_authored.jsonl if present. D24's original hard gate
("cannot begin with zero self-authored content merged in") is explicitly
lifted per D27 -- this proceeds with whatever self_authored.jsonl currently
contains, including zero rows, and always prints the exact count so that
composition is visible in every run's output rather than assumed.

Usage:
    python3 scripts/train_primary.py --success-criterion "..." \\
        [--epochs N] [--batch-size N] [--lr F] [--seed N] [--limit N]

success-criterion is required (docs/specs/DEVELOPMENT_RULES.md's TDD
exception for training runs: write down what "success" means before the run,
not after seeing the result).

--limit N caps the training set to the first N examples (after the same
seeded shuffle run_training uses internally) -- a cheap smoke test against
the real backbone/tokenizer/data before committing to a full run. This gap
(nothing between stub-model unit tests and a full ~10k-row run) is exactly
what let docs/ISSUES.md's ISSUE-1 (MPS AdamW NaN) reach a full run
undetected; use --limit for any change to the training path going forward.

ISSUE-3 (docs/ISSUES.md): every run used to write to the same hardcoded
models/primary/ with no record of which run produced what's on disk -- a
--limit smoke test silently overwrote a full run's checkpoint mid-session.
--limit runs now default to models/smoke_test/<run_id>/ instead (override
with --output-dir); every run also writes manifest.json alongside the
checkpoint recording exactly which run/config produced it, updated after
every epoch (see ISSUE-5 below), not just once at the end.

ISSUE-5 (docs/ISSUES.md, flagged by peer review): the checkpoint used to
be saved exactly once, after the entire epoch loop returned -- a failure
late in a multi-hour run (this session already hit an MPS OOM and an
AdamW NaN) would lose the whole run with nothing recoverable. Every epoch
now saves a checkpoint + manifest.json, recoverable at any interruption
point, not an all-or-nothing bet on the run finishing cleanly.

ISSUE-6 (docs/ISSUES.md, correcting ISSUE-5's first draft): each epoch
writes to its own <checkpoint_dir>/epoch_N/ subdirectory rather than all
epochs overwriting one shared path -- save_pretrained() writes several
files (config.json, model.safetensors, tokenizer files) and isn't atomic
as a whole, so an interrupted write to a shared path could corrupt the
directory while destroying the last known-good epoch's state too. Also in
this pass: torch.manual_seed() is now called before model construction
(the classifier/pooler head's random init was previously unseeded
regardless of --seed), and every checkpoint's manifest.json records which
device (mps/cuda/cpu) actually produced it.
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Portable, environment-independent: `import src...` below only works if the
# repo root is on sys.path. Locally this happened to work via an ambient
# PYTHONPATH=. set in the shell -- not something this repo's own setup docs
# ever specified, and not present in a fresh Colab shell (confirmed:
# ModuleNotFoundError: No module named 'src' when this script was first run
# there). Inserting the repo root explicitly makes `python3 scripts/*.py`
# work identically in any environment, matching every other script here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.io import load_examples_jsonl
from src.model.train import (
    ModelConfig,
    TrainingRunConfig,
    assert_self_authored_gate,
    build_model_and_tokenizer,
    epoch_checkpoint_subdir,
    resolve_checkpoint_dir,
    resolve_device,
    run_training,
)

TRAIN_PATH = Path("data/processed/train.jsonl")
SELF_AUTHORED_PATH = Path("data/processed/self_authored.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--success-criterion", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="cap the training set to N examples, for smoke-testing")
    parser.add_argument("--output-dir", type=str, default=None, help="override the checkpoint output directory")
    parser.add_argument(
        "--extra-jsonl", type=str, default=None,
        help="path to additional Example rows to merge in (e.g. data/redteam/harvested.jsonl for the harvest-retrain step, IMPLEMENTATION_PLAN.md Phase 1 DoD)",
    )
    args = parser.parse_args()

    # run_id disambiguates concurrent/repeated --limit smoke tests from each
    # other (ISSUE-5's residual gap) -- meaningless for a full run, ignored
    # there by resolve_checkpoint_dir.
    run_id = f"limit{args.limit}_lr{args.lr}_seed{args.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}" if args.limit is not None else None
    checkpoint_dir = resolve_checkpoint_dir(args.output_dir, args.limit, run_id=run_id)

    curated = load_examples_jsonl(TRAIN_PATH)
    self_authored = load_examples_jsonl(SELF_AUTHORED_PATH)
    extra = load_examples_jsonl(args.extra_jsonl) if args.extra_jsonl else []
    assert curated, f"{TRAIN_PATH} is missing or empty -- run scripts/split_curated_pool.py first"
    assert_self_authored_gate(self_authored)

    examples = curated + self_authored + extra
    if args.limit is not None:
        shuffled = examples[:]
        random.Random(args.seed).shuffle(shuffled)
        examples = shuffled[: args.limit]
        print(f"--limit {args.limit}: smoke-test run, not a full training pass -> {checkpoint_dir}")
    extra_note = f" + {len(extra)} from {args.extra_jsonl}" if args.extra_jsonl else ""
    print(f"training set: {len(curated)} curated + {len(self_authored)} self-authored{extra_note}, {len(examples)} used this run")

    model_config = ModelConfig()  # D13: primary backbone = microsoft/deberta-v3-small
    run_config = TrainingRunConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        success_criterion=args.success_criterion,
    )

    # Peer-review finding: torch's global RNG was never seeded anywhere in
    # this codebase -- the classifier/pooler head's random init came from
    # whatever state torch's RNG happened to be in, unreproducible run to
    # run even with the same --seed. Seeding here, immediately before
    # build_model_and_tokenizer constructs the model, is what actually
    # makes run_config.seed cover the whole run, not just batch order.
    device = resolve_device(model_config)
    print(f"resolved device: {device}")
    model, tokenizer = build_model_and_tokenizer(model_config, seed=args.seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(epoch: int, loss_history_so_far):
        # Each epoch gets its own subdirectory, never overwriting a prior
        # epoch's -- save_pretrained() isn't atomic (writes config.json,
        # model.safetensors, tokenizer files separately); an interrupted
        # write to a shared path could corrupt it while destroying the last
        # known-good epoch's state (docs/ISSUES.md ISSUE-6 correction).
        epoch_dir = epoch_checkpoint_subdir(checkpoint_dir, epoch)
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir, safe_serialization=True)
        tokenizer.save_pretrained(epoch_dir)
        manifest = {
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "completed_epochs": epoch + 1,
            "total_epochs": args.epochs,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "adam_eps": run_config.adam_eps,
            "seed": args.seed,
            "n_curated": len(curated),
            "n_self_authored": len(self_authored),
            "extra_jsonl": args.extra_jsonl,
            "n_extra": len(extra),
            "n_examples_used": len(examples),
            "loss_history": loss_history_so_far,
            "success_criterion": run_config.success_criterion,
        }
        with open(epoch_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  epoch {epoch + 1}/{args.epochs} done, loss={loss_history_so_far[-1]:.4f} -- checkpoint saved to {epoch_dir}")
        return epoch_dir

    epoch_loss_history = []
    last_epoch_dir = [None]  # mutable cell so on_epoch_end's closure can update it

    def on_epoch_end(epoch: int, mean_loss: float) -> None:
        epoch_loss_history.append(mean_loss)
        last_epoch_dir[0] = save_checkpoint(epoch, epoch_loss_history)

    result = run_training(model, tokenizer, examples, model_config, run_config, on_epoch_end=on_epoch_end)

    print(f"loss history: {result.loss_history}")
    print(f"success criterion: {run_config.success_criterion}")
    print(f"self-authored examples merged into this run: {len(self_authored)}")
    print(f"final checkpoint + manifest.json saved to {last_epoch_dir[0]}")


if __name__ == "__main__":
    main()
