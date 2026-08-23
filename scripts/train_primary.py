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
now saves a checkpoint + manifest.json to the same output dir, overwriting
the previous epoch's save -- the dir always holds the latest complete
epoch's weights, recoverable at any interruption point, not an
all-or-nothing bet on the run finishing cleanly.
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from src.data.io import load_examples_jsonl
from src.model.train import (
    ModelConfig,
    TrainingRunConfig,
    assert_self_authored_gate,
    build_model_and_tokenizer,
    resolve_checkpoint_dir,
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
    args = parser.parse_args()

    # run_id disambiguates concurrent/repeated --limit smoke tests from each
    # other (ISSUE-5's residual gap) -- meaningless for a full run, ignored
    # there by resolve_checkpoint_dir.
    run_id = f"limit{args.limit}_lr{args.lr}_seed{args.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}" if args.limit is not None else None
    checkpoint_dir = resolve_checkpoint_dir(args.output_dir, args.limit, run_id=run_id)

    curated = load_examples_jsonl(TRAIN_PATH)
    self_authored = load_examples_jsonl(SELF_AUTHORED_PATH)
    assert curated, f"{TRAIN_PATH} is missing or empty -- run scripts/split_curated_pool.py first"
    assert_self_authored_gate(self_authored)

    examples = curated + self_authored
    if args.limit is not None:
        shuffled = examples[:]
        random.Random(args.seed).shuffle(shuffled)
        examples = shuffled[: args.limit]
        print(f"--limit {args.limit}: smoke-test run, not a full training pass -> {checkpoint_dir}")
    print(f"training set: {len(curated)} curated + {len(self_authored)} self-authored, {len(examples)} used this run")

    model_config = ModelConfig()  # D13: primary backbone = microsoft/deberta-v3-small
    run_config = TrainingRunConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        success_criterion=args.success_criterion,
    )

    model, tokenizer = build_model_and_tokenizer(model_config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(epoch: int, loss_history_so_far):
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_dir)
        manifest = {
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_epochs": epoch + 1,
            "total_epochs": args.epochs,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "adam_eps": run_config.adam_eps,
            "seed": args.seed,
            "n_curated": len(curated),
            "n_self_authored": len(self_authored),
            "n_examples_used": len(examples),
            "loss_history": loss_history_so_far,
            "success_criterion": run_config.success_criterion,
        }
        with open(checkpoint_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  epoch {epoch + 1}/{args.epochs} done, loss={loss_history_so_far[-1]:.4f} -- checkpoint saved to {checkpoint_dir}")

    epoch_loss_history = []

    def on_epoch_end(epoch: int, mean_loss: float) -> None:
        epoch_loss_history.append(mean_loss)
        save_checkpoint(epoch, epoch_loss_history)

    result = run_training(model, tokenizer, examples, model_config, run_config, on_epoch_end=on_epoch_end)

    print(f"loss history: {result.loss_history}")
    print(f"success criterion: {run_config.success_criterion}")
    print(f"self-authored examples merged into this run: {len(self_authored)}")
    print(f"final checkpoint + manifest.json saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
