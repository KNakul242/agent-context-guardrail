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
    python3 scripts/run_redteam.py --checkpoint models/primary
"""

import argparse
import json

from src.data.io import write_examples_jsonl
from src.model.inference import predict_label_fn
from src.model.train import ModelConfig, build_model_and_tokenizer
from src.redteam.harness import bypass_rate_by_technique, harvest_bypasses, run_seeds
from src.redteam.seeds import DEFAULT_SEEDS_PATH, load_seeds

HARVEST_PATH = "data/redteam/harvested.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to a saved model+tokenizer directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    seeds = load_seeds()
    assert seeds, f"no red-team seeds found at {DEFAULT_SEEDS_PATH}"

    config = ModelConfig(backbone=args.checkpoint)
    model, tokenizer = build_model_and_tokenizer(config)
    predict_fn = predict_label_fn(model, tokenizer, config, threshold=args.threshold)

    results = run_seeds(seeds, predict_fn)
    rates = bypass_rate_by_technique(results)
    print(json.dumps(rates, indent=2))

    harvested = harvest_bypasses(results)
    write_examples_jsonl(HARVEST_PATH, harvested)
    print(f"\nharvested {len(harvested)} / {len(seeds)} confirmed bypasses -> {HARVEST_PATH}")


if __name__ == "__main__":
    main()
