#!/usr/bin/env python3
"""
In-distribution eval report for a trained checkpoint (Phase 1 DoD:
F1, ROC-AUC, recall@1%-FPR, FPR on the hard-negative benign subset).

Hard-negative subset proxy: source == "notinject". D19 established NotInject
IS the hard-negative class by construction (339 benign prompts containing
trigger words with no actual injection) -- other sources' benign rows are
never counted here even if some happen to contain similar trigger language,
since only NotInject's construction methodology actually verifies that
property row-by-row.

Usage:
    python3 scripts/evaluate.py --checkpoint models/primary --split data/processed/val.jsonl
"""

import argparse
import json

from src.data.io import load_examples_jsonl
from src.data.schema import Label
from src.eval.report import build_eval_report
from src.model.inference import predict_scores
from src.model.train import ModelConfig, build_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to a saved model+tokenizer directory")
    parser.add_argument("--split", default="data/processed/val.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    examples = load_examples_jsonl(args.split)
    assert examples, f"{args.split} is missing or empty"

    config = ModelConfig(backbone=args.checkpoint)
    model, tokenizer = build_model_and_tokenizer(config)

    y_true = [1 if ex.label == Label.MALICIOUS else 0 for ex in examples]
    y_scores = predict_scores(model, tokenizer, examples, config)
    is_hard_negative = [ex.source == "notinject" for ex in examples]

    report = build_eval_report(y_true, y_scores, is_hard_negative, threshold=args.threshold)
    report["checkpoint"] = args.checkpoint
    report["split"] = args.split

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
