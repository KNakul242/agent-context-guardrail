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
    python3 scripts/escalate_redteam.py --checkpoint models/primary/epoch_3 [--max-rounds N]
        [--threshold F | --derive-threshold-from-val PATH [--max-fpr F]]

--threshold defaults to 0.5, which has no connection to the recall@1%-FPR
operating point scripts/evaluate.py's headline metric describes (D6) --
a red-team bypass rate measured at an unrelated decision boundary answers
a different question than the eval report, which matters directly for
D7's ModernBERT gate (peer-review finding). Pass
--derive-threshold-from-val data/processed/val.jsonl to instead compute
the actual score cutoff that achieves --max-fpr (default 0.01) on that
split, via src.eval.metrics.threshold_at_fpr, and red-team at that exact
operating point instead of an arbitrary default.
"""

import argparse
import json
import sys
from collections import Counter
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
from src.redteam.harness import harvest_bypasses, run_escalation, seeds_exceeding_max_length
from src.redteam.llm_client import build_gemini_llm_call
from src.redteam.seeds import DEFAULT_SEEDS_PATH, load_seeds

HARVEST_PATH = Path("data/redteam/harvested_escalation.jsonl")


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
    parser.add_argument("--max-rounds", type=int, default=5)
    args = parser.parse_args()

    seeds = load_seeds()
    assert seeds, f"no red-team seeds found at {DEFAULT_SEEDS_PATH}"

    config = ModelConfig(backbone=args.checkpoint)
    model, tokenizer = build_model_and_tokenizer(config)
    threshold = resolve_threshold(config, model, tokenizer, args)
    predict_fn = predict_label_fn(model, tokenizer, config, threshold=threshold)
    llm_call = build_gemini_llm_call()

    # ISSUE-9 (docs/ISSUES.md): predict_label_fn always right-truncates --
    # a seed whose real tokenized length exceeds max_length can have its
    # payload silently cut off before scoring, and every round-N mutation
    # this loop generates inherits the same risk. Flag base seeds now,
    # loudly, so this isn't silently repeated across escalation rounds too.
    token_counts = {seed.seed_id: len(tokenizer(seed.content)["input_ids"]) for seed in seeds}
    truncated_seed_ids = seeds_exceeding_max_length(token_counts, max_length=config.max_length)
    if truncated_seed_ids:
        affected_techniques = sorted({s.technique.value for s in seeds if s.seed_id in truncated_seed_ids})
        print(
            f"WARNING (ISSUE-9): {len(truncated_seed_ids)} base seed(s) exceed max_length={config.max_length} "
            f"tokens and will be right-truncated before scoring -- results for these (and any escalated "
            f"mutations of them) are NOT reliable evidence of model behavior: {truncated_seed_ids}\n"
            f"  affected techniques: {affected_techniques}\n"
        )

    # Incremental persistence (peer-review finding): write harvested
    # bypasses to disk as each seed's escalation completes, not only after
    # the full corpus finishes -- a live API failure late in a long run
    # shouldn't cost every already-computed seed's results, same rationale
    # as ISSUE-5's per-epoch training checkpoints.
    harvested_so_far = []

    def on_seed_done(seed_id: str, trajectory) -> None:
        harvested_so_far.extend(harvest_bypasses(trajectory))
        write_examples_jsonl(HARVEST_PATH, harvested_so_far)

    trajectories = run_escalation(seeds, predict_fn, llm_call, max_rounds=args.max_rounds, on_seed_done=on_seed_done)

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
    print(f"\nharvested {len(harvested_so_far)} confirmed bypasses (base + escalated) -> {HARVEST_PATH}")


if __name__ == "__main__":
    main()
