#!/usr/bin/env python3
"""
Runs the D10 manual seed corpus (data/redteam/seeds.jsonl) against a trained
checkpoint. Reports bypass rate segmented by InjectionTechnique -- never one
aggregate ASR number (D10) -- and writes every confirmed bypass to
data/redteam/harvested.jsonl for the harvest-retrain step
(docs/specs/IMPLEMENTATION_PLAN.md Phase 1).

The automated LLM-red-teamer escalation loop (src/redteam/harness.py's
escalate()) is not wired in here -- it's still the documented Phase 1 stub,
unimplemented until there's a trained model to mutate seeds against, which
this script is what produces the results that loop consumes next.

Usage:
    python3 scripts/run_redteam.py --checkpoint models/primary/epoch_3
        [--threshold F | --derive-threshold-from-val PATH [--max-fpr F]]

--threshold defaults to 0.5, which has no connection to the recall@1%-FPR
operating point scripts/evaluate.py's headline metric describes (D6) --
a bypass rate measured at an unrelated decision boundary answers a
different question than the eval report, which matters directly for D7's
ModernBERT gate (peer-review finding). Pass --derive-threshold-from-val
data/processed/val.jsonl to instead compute the actual score cutoff that
achieves --max-fpr (default 0.01) on that split, via
src.eval.metrics.threshold_at_fpr, and red-team at that exact operating
point instead of an arbitrary default.
"""

import argparse
import json
import sys
from pathlib import Path

# See scripts/train_primary.py's identical bootstrap for why this is needed:
# `import src...` requires the repo root on sys.path, which was only true
# locally via an ambient (undocumented) PYTHONPATH=., not in a fresh Colab shell.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.io import load_examples_jsonl, write_examples_jsonl
from src.data.schema import Label
from src.eval.metrics import threshold_at_fpr
from src.model.inference import predict_label_fn, predict_scores
from src.model.train import ModelConfig, build_model_and_tokenizer
from src.redteam.harness import bypass_rate_by_technique, harvest_bypasses, run_seeds
from src.redteam.seeds import DEFAULT_SEEDS_PATH, load_seeds

HARVEST_PATH = "data/redteam/harvested.jsonl"


def resolve_threshold(config: ModelConfig, model, tokenizer, args) -> float:
    if args.derive_threshold_from_val is None:
        return args.threshold
    val_examples = load_examples_jsonl(args.derive_threshold_from_val)
    assert val_examples, f"{args.derive_threshold_from_val} is missing or empty"
    y_true = [1 if ex.label == Label.MALICIOUS else 0 for ex in val_examples]
    y_scores = predict_scores(model, tokenizer, val_examples, config)
    threshold = threshold_at_fpr(y_true, y_scores, max_fpr=args.max_fpr)
    print(f"derived threshold={threshold:.4f} from {args.derive_threshold_from_val} at max_fpr={args.max_fpr}")
    return threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to a saved model+tokenizer directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--derive-threshold-from-val", type=str, default=None, help="path to a val split; overrides --threshold")
    parser.add_argument("--max-fpr", type=float, default=0.01)
    args = parser.parse_args()

    seeds = load_seeds()
    assert seeds, f"no red-team seeds found at {DEFAULT_SEEDS_PATH}"

    config = ModelConfig(backbone=args.checkpoint)
    model, tokenizer = build_model_and_tokenizer(config)
    threshold = resolve_threshold(config, model, tokenizer, args)
    predict_fn = predict_label_fn(model, tokenizer, config, threshold=threshold)

    results = run_seeds(seeds, predict_fn)
    rates = bypass_rate_by_technique(results)
    print(json.dumps(rates, indent=2))

    harvested = harvest_bypasses(results)
    write_examples_jsonl(HARVEST_PATH, harvested)
    print(f"\nharvested {len(harvested)} / {len(seeds)} confirmed bypasses -> {HARVEST_PATH}")


if __name__ == "__main__":
    main()
