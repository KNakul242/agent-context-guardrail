# results/

Persisted output of `scripts/evaluate.py` and `scripts/run_redteam.py`, regenerated
directly from the checkpoints in this repo/on HF and re-verified against the
numbers already reported in `docs/ISSUES.md`/`docs/DECISIONS.md` before being
saved here. Every file below is a deterministic re-run (fixed checkpoint + fixed
data + fixed seed), not a reconstruction from memory — see each entry for the
exact command and what it was checked against.

## In-distribution eval (`scripts/evaluate.py`, `data/processed/val.jsonl`)

- `eval_primary_epoch3.json` — `models/primary/epoch_3` (the primary/submitted
  model, local MPS run). Matches D31's "base (epoch_3)" row exactly: F1 0.968,
  ROC-AUC 0.995, recall@1%FPR 0.950, hard-negative FPR 0.088 (3/34).
- `eval_colab_epoch3.json` — `models/colab_primary_epoch3_recovered/checkpoint/epoch_3`,
  the Colab-CUDA recovery copy of the same training run (D29: local and Colab ran
  in parallel, local's checkpoint was used as primary). Numbers differ slightly
  from the local run (F1 0.972 vs 0.968, hard-negative FPR 1/34 vs 3/34) — expected,
  not a bug: same recipe/data/seed but a different backend (MPS vs CUDA) produces
  a numerically different, independently-trained model, not a byte-identical copy.
  Never separately reported as an official metric before this file; included for
  completeness since the checkpoint exists in the repo.
- `eval_retrained_epoch2.json` — `models/primary_retrained/epoch_2`, the D31
  harvest-retrain experiment. Matches D31's "retrained (warm-start, epoch_2)" row
  exactly: F1 0.961, ROC-AUC 0.992, recall@1%FPR 0.936, hard-negative FPR 0.088
  (3/34, unchanged).

## Red-team base pass (`scripts/run_redteam.py`, 35-seed corpus, `--derive-threshold-from-val`)

- `redteam_base_epoch3.json` — `models/primary/epoch_3`. Derived threshold 0.7413,
  matching ISSUE-9/ISSUE-10's documented operating point exactly; per-technique
  bypass rates match `docs/ISSUES.md` exactly. Running this also regenerates
  `data/redteam/harvested.jsonl` — confirmed byte-identical to the already-committed
  file (`git diff` empty) before this was saved.
- `redteam_base_retrained_epoch2.json` — `models/primary_retrained/epoch_2`.
  Derived threshold 0.4853, matching D31 exactly; `low_resource_language` bypass
  rate 0.0% (0/8), matching D31's reported "62.5% → 0%" result exactly.
  **Caveat:** running this script against a second checkpoint also overwrites
  `data/redteam/harvested.jsonl` with *this* model's harvest (its own bypass set,
  not the primary model's) — that overwrite was reverted via `git checkout` after
  generating this file, so the committed `harvested.jsonl` still reflects the
  primary model's (`epoch_3`) harvest, as documented. This script does not
  currently support a non-default harvest output path; if it's ever run again
  standalone for a second checkpoint, re-check `git status` on
  `data/redteam/harvested.jsonl` afterward.

## Escalation loop (not regenerated — see below)

The adversarial escalation pass (`src/redteam/harness.py::escalate()`, live
Gemini calls, non-deterministic) is **not** reproducible on demand — re-running
it would produce different mutations, not the documented result. Faking or
reconstructing matching numbers here would misrepresent a non-deterministic
process as a saved artifact. What's real and inspectable instead:

- `data/redteam/harvested_escalation.jsonl` (12 rows) — the base+escalated
  harvest from the escalation run reported in `docs/ISSUES.md` ISSUE-10, already
  committed to the repo at that path (not duplicated here).
- `data/redteam/harvested_merged.jsonl` (15 rows) — that harvest merged with 3
  additional `low_resource_language` bypasses found in a follow-up pass
  (`docs/DECISIONS.md` D31/D32), also already committed at that path.

The escalation run's own session transcript/log was not preserved separately —
these two files are the real, persisted evidence of what was found, not a
summary standing in for a missing log.
