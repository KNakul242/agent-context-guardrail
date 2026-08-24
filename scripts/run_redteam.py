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

ISSUE-9 (docs/ISSUES.md): predict_label_fn always right-truncates content
before scoring -- deliberately, not a bug (see src/model/inference.py's
docstring): a real deployed guardrail has no oracle at inference time
telling it where a hidden payload sits, so right-truncation is arguably
MORE representative of production reality than always preserving the
payload would be. But for seeds whose content exceeds max_length, this
means a "bypass" can mean either "the model saw the payload and missed
it" (in_window, a genuine capability gap) or "the payload was truncated
away before the model ever ran" (out_of_window, zero signal reached the
classifier -- not fixed by retraining, only by a different scoring
strategy). For any seed with an injection_marker registered
(data/redteam/seeds.jsonl), this script reports both components
separately rather than blending them into one number.
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
from src.redteam.harness import (
    bypass_rate_by_technique,
    harvest_bypasses,
    marker_token_offset,
    run_seeds,
    seeds_exceeding_max_length,
    split_bypass_by_truncation_window,
)
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
    parser.add_argument("--output", type=str, default=None, help="also write a JSON summary (threshold, per-technique rates, truncation split, harvest count) to this path")
    args = parser.parse_args()

    seeds = load_seeds()
    assert seeds, f"no red-team seeds found at {DEFAULT_SEEDS_PATH}"

    config = ModelConfig(backbone=args.checkpoint)
    model, tokenizer = build_model_and_tokenizer(config)
    threshold = resolve_threshold(config, model, tokenizer, args)
    predict_fn = predict_label_fn(model, tokenizer, config, threshold=threshold)

    # ISSUE-9 (docs/ISSUES.md): general safety net -- flag ANY seed whose
    # real tokenized length exceeds max_length, even one with no
    # injection_marker registered (the split report below only covers
    # marker-tagged seeds).
    token_counts = {seed.seed_id: len(tokenizer(seed.content)["input_ids"]) for seed in seeds}
    truncated_seed_ids = seeds_exceeding_max_length(token_counts, max_length=config.max_length)
    if truncated_seed_ids:
        affected_techniques = sorted({s.technique.value for s in seeds if s.seed_id in truncated_seed_ids})
        print(
            f"NOTE (ISSUE-9): {len(truncated_seed_ids)} seed(s) exceed max_length={config.max_length} "
            f"tokens and will be right-truncated before scoring: {truncated_seed_ids}\n"
            f"  affected techniques: {affected_techniques}\n"
        )

    results = run_seeds(seeds, predict_fn)
    rates = bypass_rate_by_technique(results)
    print("bypass rate by technique (blended -- see the in/out-of-window split below for any technique with registered injection_markers):")
    print(json.dumps(rates, indent=2))

    # ISSUE-9 split report (peer-review refinement): for any seed with a
    # known injection_marker, separate "model saw the payload and missed
    # it" from "payload was truncated away before scoring" -- these are
    # different findings with different fixes, and out_of_window is
    # arguably stronger evidence for D7's context-length gate than a
    # clean in_window miss, not weaker or discardable.
    count_tokens = lambda s: len(tokenizer(s)["input_ids"])
    marker_offsets = {
        seed.seed_id: marker_token_offset(seed.content, seed.injection_marker, count_tokens)
        for seed in seeds
        if seed.injection_marker is not None
    }
    marker_offsets = {k: v for k, v in marker_offsets.items() if v is not None}
    out_of_window_seed_ids = set()
    if marker_offsets:
        split = split_bypass_by_truncation_window(results, marker_offsets, max_length=config.max_length)
        print("\ntruncation-window split (marker-tagged seeds only, ISSUE-9):")
        print(json.dumps(split, indent=2))
        out_of_window_seed_ids = {sid for sid, offset in marker_offsets.items() if offset >= config.max_length}

    # ISSUE-9 harvest fix (peer review, second pass): an out-of-window
    # "bypass" isn't a real missed attack -- the payload never reached the
    # classifier, so the row would be mislabeled MALICIOUS training data
    # for something the model was never actually shown. Never harvest these.
    harvestable_results = [r for r in results if r.seed.seed_id not in out_of_window_seed_ids]
    excluded_count = len(results) - len(harvestable_results)
    if excluded_count:
        print(f"\n({excluded_count} out-of-window result(s) excluded from harvesting -- truncation artifacts, not genuine bypasses, per ISSUE-9)")

    harvested = harvest_bypasses(harvestable_results)
    write_examples_jsonl(HARVEST_PATH, harvested)
    print(f"\nharvested {len(harvested)} / {len(seeds)} confirmed bypasses -> {HARVEST_PATH}")

    if args.output:
        summary = {
            "checkpoint": args.checkpoint,
            "threshold": threshold,
            "n_seeds": len(seeds),
            "bypass_rate_by_technique": rates,
            "truncation_window_split": split if marker_offsets else None,
            "n_harvested": len(harvested),
            "harvest_path": HARVEST_PATH,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"summary written to {args.output}")


if __name__ == "__main__":
    main()
