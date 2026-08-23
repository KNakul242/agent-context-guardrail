#!/usr/bin/env python3
"""
D10's automated escalation loop: mutates every seed in data/redteam/seeds.jsonl
that a trained checkpoint currently catches (Maatphor-style, via a live
Gemini call) until each either bypasses or exhausts its per-seed round
budget. Reports the per-technique escalation outcome and writes every
confirmed bypass (base seed or mutated) to
data/redteam/harvested_escalation.jsonl for the harvest-retrain step.

Kept separate from scripts/run_redteam.py rather than folding escalation
into it: the base run is a single deterministic pass costing zero live API
calls; this script costs one live call per unsuccessful round per caught
seed, and that's worth an explicit, separate invocation rather than an
implicit cost added to every red-team run.

Usage:
    python3 scripts/escalate_redteam.py --checkpoint models/primary [--max-rounds N]
"""

import argparse
import json
from collections import Counter

from src.data.io import write_examples_jsonl
from src.model.inference import predict_label_fn
from src.model.train import ModelConfig, build_model_and_tokenizer
from src.redteam.harness import harvest_bypasses, run_escalation
from src.redteam.llm_client import build_gemini_llm_call
from src.redteam.seeds import DEFAULT_SEEDS_PATH, load_seeds

HARVEST_PATH = "data/redteam/harvested_escalation.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to a saved model+tokenizer directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-rounds", type=int, default=5)
    args = parser.parse_args()

    seeds = load_seeds()
    assert seeds, f"no red-team seeds found at {DEFAULT_SEEDS_PATH}"

    config = ModelConfig(backbone=args.checkpoint)
    model, tokenizer = build_model_and_tokenizer(config)
    predict_fn = predict_label_fn(model, tokenizer, config, threshold=args.threshold)
    llm_call = build_gemini_llm_call()

    trajectories = run_escalation(seeds, predict_fn, llm_call, max_rounds=args.max_rounds)

    all_results = [r for traj in trajectories.values() for r in traj]
    per_technique_outcome = Counter()
    for seed in seeds:
        traj = trajectories[seed.seed_id]
        outcome = "bypassed_immediately" if len(traj) == 1 and traj[-1].bypassed else (
            "bypassed_after_escalation" if traj[-1].bypassed else "never_bypassed"
        )
        per_technique_outcome[(seed.technique.value, outcome)] += 1
        print(f"{seed.seed_id} ({seed.technique.value}): {outcome} in {len(traj)} round(s)")

    print("\nper-technique escalation outcome:")
    print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in sorted(per_technique_outcome.items())}, indent=2))

    harvested = harvest_bypasses(all_results)
    write_examples_jsonl(HARVEST_PATH, harvested)
    print(f"\nharvested {len(harvested)} confirmed bypasses (base + escalated) -> {HARVEST_PATH}")


if __name__ == "__main__":
    main()
