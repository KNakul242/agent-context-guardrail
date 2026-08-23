#!/usr/bin/env python3
"""
Real training entrypoint for the primary detector (Phase 1,
docs/specs/IMPLEMENTATION_PLAN.md; D13 backbone resolved, D24 self-authored
gate).

Loads data/processed/train.jsonl (the curated public-source pool) plus
data/processed/self_authored.jsonl (Track 0B, merged in whenever produced),
and refuses to proceed -- loudly, via assert_self_authored_gate() in
src/model/train.py -- if the self-authored file is missing or empty. This is
D24's hard gate made unskippable rather than a convention to remember: "a
real training run may begin once at least some ... self-authored/adapted
examples are present ... it cannot begin with zero self-authored content
merged in, regardless of time remaining."

Usage:
    python3 scripts/train_primary.py --success-criterion "..." \\
        [--epochs N] [--batch-size N] [--lr F] [--seed N]

success-criterion is required (docs/specs/DEVELOPMENT_RULES.md's TDD
exception for training runs: write down what "success" means before the run,
not after seeing the result).
"""

import argparse
from pathlib import Path

from src.data.io import load_examples_jsonl
from src.model.train import (
    ModelConfig,
    TrainingRunConfig,
    assert_self_authored_gate,
    build_model_and_tokenizer,
    run_training,
)

TRAIN_PATH = Path("data/processed/train.jsonl")
SELF_AUTHORED_PATH = Path("data/processed/self_authored.jsonl")
CHECKPOINT_DIR = Path("models/primary")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--success-criterion", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    curated = load_examples_jsonl(TRAIN_PATH)
    self_authored = load_examples_jsonl(SELF_AUTHORED_PATH)
    assert curated, f"{TRAIN_PATH} is missing or empty -- run scripts/split_curated_pool.py first"
    assert_self_authored_gate(self_authored)

    examples = curated + self_authored
    print(f"training set: {len(curated)} curated + {len(self_authored)} self-authored = {len(examples)} total")

    model_config = ModelConfig()  # D13: primary backbone = microsoft/deberta-v3-small
    run_config = TrainingRunConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        success_criterion=args.success_criterion,
    )

    model, tokenizer = build_model_and_tokenizer(model_config)
    result = run_training(model, tokenizer, examples, model_config, run_config)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    print(f"loss history: {result.loss_history}")
    print(f"success criterion: {run_config.success_criterion}")
    print(f"self-authored examples merged into this run: {len(self_authored)}")
    print(f"checkpoint saved to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
