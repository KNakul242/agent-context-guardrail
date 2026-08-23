# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An assignment (`docs/specs/PROBLEM_STATEMENT.md`, verbatim, ground
truth — the original brief names the assigning company; per
`docs/specs/VISION.md` that name is deliberately not repeated elsewhere in
this repo). The chosen track: train a detector for **indirect prompt
injection (IPI)** hidden in tool outputs (files, webpages, API responses)
that an LLM agent reads mid-task. Malicious/dangerous tool-call detection is
an in-scope stretch extension, built on the same schema, only after the IPI
pipeline is done.

**Read the docs before writing code — they are not background reading, they
govern how work in this repo is supposed to proceed:**
- `docs/specs/VISION.md` — reasoning posture (first-principles-outward,
  second-order thinking, Occam's razor) that should shape every non-trivial
  decision, plus the operating persona and communication style for this repo
  (see below). Read before starting each new phase.
- `docs/DECISIONS.md` — running decision log with rationale, alternatives
  considered, and why rejected. Source of truth for *why* the repo is shaped
  the way it is. Amend it, don't silently drift from it.
- `docs/specs/DEVELOPMENT_RULES.md` — governs *how* code gets written/merged
  (git workflow, TDD policy, code style, Definition of Done).
- `docs/specs/IMPLEMENTATION_PLAN.md` — phase-by-phase build guide, one
  branch per phase, each with its own Definition of Done.
- `docs/specs/PROBLEM_STATEMENT.md` — the original assignment, verbatim. If
  anything else conflicts with this, this wins.

If a logged decision or plan step doesn't hold up against what you find
while implementing, **stop and flag it as a proposed amendment before
proceeding** — do not silently build a workaround. See "Flagging drift" in
`DEVELOPMENT_RULES.md`.

## Current state

**Phase 0 / Track 0A is closed and merged to `develop`** (merge commit
`2ba3a2f`, feature branch `feature/data-foundation-public` at `7ed4bd1`).
Working branch is now `develop`. All four settled-pool public sources
pulled/schema-mapped/validated (`src/data/sources/{notinject,prodnull,malmasabi,bipia}.py`),
eval metrics (`src/eval/metrics.py`), the D8a aggregation module
(`src/model/aggregation.py`), the red-team harness skeleton
(`src/redteam/harness.py`), and the training script scaffold
(`src/model/train.py`, config-driven backbone) all exist with passing
tests (`tests/`, 88 tests, `pytest`).

**D13 is resolved: primary backbone is DeBERTa-v3-small (142M)** (see
`docs/DECISIONS.md` D13). Composition pass complete — the raw 117,460-row
bulk pool was curated down to `data/processed/curated_pool.jsonl` (13,032
rows, exactly 6,516 malicious / 6,516 benign), via `src/data/curate.py`'s
stratified-sampling/coherence-filter functions, reproducibly regenerable
via `scripts/build_curated_pool.py`. Full derivation, per-domain
breakdowns, and the license-provenance finding that drove excluding
MAlmasabi's malicious half entirely are in `docs/data_summary.md` §0
(supersedes §1-9's raw-pool numbers as the actual dataset going forward).

**The real train/val/test split is materialized**:
`data/processed/{train,val,test}.jsonl` (10,430 / 1,293 / 1,309 rows),
via `scripts/split_curated_pool.py`, using D25's document-group-aware
stratified split (BIPIA rows grouped by `(task, clean_context_id)` so a
matched benign/malicious pair can't span two splits) — verified 0 leaks
across 901 BIPIA document groups.

**Not yet done, explicitly non-blocking for Phase 0's closed DoD, per
D24:** Track 0B (self-authored examples — user-led;
`scripts/example_lab_gemini.py` exists, output not yet produced). It
merges into the curated pool opportunistically, whenever produced. No
real training run and no Phase 1 completion until it lands — that
constraint is still live (D24: a training run needs *some*, not zero,
self-authored content merged in) — but it does not gate this already-closed
branch.

**Phase 1 (`feature/primary-detector-redteam`, branched from `develop`) is
in progress.** Built so far, all with passing tests (`tests/`, 108 tests):
- `src/data/io.py` — shared JSONL read/write for `Example` (factored out of
  the two Phase 0 build scripts, which each hand-rolled it).
- `data/redteam/seeds.jsonl` (27 hand-authored seeds, 3 per
  `InjectionTechnique` except `OTHER`) + `src/redteam/seeds.py::load_seeds()`
  — D10's manual-seed layer. `PAYLOAD_SPLIT` seeds split the payload across
  fields of one document, not across turns — see D26 for why that's
  in-scope despite D8a's window=1 limitation.
- `src/model/inference.py` — `predict_scores()` (batched P(malicious)) and
  `predict_label_fn()` (the `str -> Label` closure `src/redteam/harness.py`'s
  `PredictFn` needs).
- `src/eval/report.py::build_eval_report()` — assembles the Phase 1 DoD's
  four in-distribution metrics (F1, ROC-AUC, recall@1%-FPR, hard-negative
  FPR) from plain score/label lists.
- Three real entrypoints, each smoke-tested end-to-end against a tiny stub
  checkpoint (not a real model — wiring proof only): `scripts/train_primary.py`,
  `scripts/evaluate.py`, `scripts/run_redteam.py`.

**`scripts/train_primary.py` currently refuses to run** — verified by
actually running it, not just by inspection — because
`data/processed/self_authored.jsonl` doesn't exist yet.
`assert_self_authored_gate()` in `src/model/train.py` enforces D24's hard
gate as code, not just as a rule to remember: zero self-authored examples
merged in is a hard stop, regardless of instruction to "move ahead." This
is intentional and does not change until Track 0B produces at least one
example. Once it does, Phase 1's actual training run, eval report, and
red-team run (against a real checkpoint, not the stub used for wiring
smoke tests above) are the next steps, per
`docs/specs/IMPLEMENTATION_PLAN.md` Phase 1's DoD — none of that is done
yet, only the wiring to do it is.

`data/raw/{notinject,prodnull,malmasabi,bipia}/` hold each source's
untouched pull (gitignored, regenerate via each source module's
`pull_raw()`). `data/processed/*` is also gitignored (only `.gitkeep`
tracked) — `curated_pool.jsonl` and the split files are regenerable
artifacts, not committed. `third_party/BIPIA` is a gitignored clone —
only its `bipia.data.utils` insertion primitives are reused, not its
builder classes or dependency list (see `bipia.py`'s docstring,
`docs/CITATIONS.md` C7).

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tests (pytest; all deterministic code per DEVELOPMENT_RULES.md's TDD scope)
python3 -m pytest tests/ -q
```

No lint config or build step exists yet.

Two execution environments, used deliberately:
- **Local (Mac, MPS backend)** — all classifier training/eval. DeBERTa-v3-small
  trains in minutes on MPS.
- **Colab (free tier)** — reserved for calling a stronger free-tier model
  (OpenRouter) for synthetic data generation and the LM-judge baseline.

## Architecture

```
data/
  raw/            untouched pulls from public sources (see docs/DATA_SOURCES.md
                   for the verified pool: prodnull, NotInject, MAlmasabi, BIPIA)
  processed/      cleaned, labeled, schema-conformant, split train/val/test
  redteam/        adversarial examples — seeds, harvested bypasses, mutation-loop outputs
src/
  data/           dataset construction, cleaning, schema (schema.py), license-check helpers
  model/          encoder classifier: model definition, aggregation module (D8a), train.py
  eval/           metrics: F1/AUC, recall@low-FPR, hard-negative FPR, red-team ASR by category
  redteam/        seed attacks, LLM-red-teamer escalation loop, harvest-to-training-set logic
notebooks/        exploratory only — nothing load-bearing lives only in a notebook
models/           trained checkpoints (also mirrored to HF Hub)
```

### Core schema (`src/data/schema.py`)

One `Example` dataclass is the unit of data across the whole pipeline
(construction, training, eval, red-team). Two fields exist specifically to
avoid future restructuring, per D8:
- `content_source_type: ContentSourceType` (`TOOL_OUTPUT` | `TOOL_CALL`) —
  lets the malicious-tool-call stretch surface (D9) reuse this schema with
  no migration.
- `candidate_content: list[str]` — a *window* of recent tool outputs, not a
  single string, default window size 1. Supports future multi-turn context
  aggregation (D8a) without touching the schema or data pipeline.

**D21 — `Label` is deliberately overloaded, read the docstring before
touching D9:** `MALICIOUS` means "content contains a hijack instruction"
for `TOOL_OUTPUT` examples, but "action is dangerous to execute" for
`TOOL_CALL` examples — two different classification tasks sharing a schema
for engineering convenience. If D9 is ever built, train it as a separate
model/head; never pool the two into one decision boundary.
`InjectionTechnique` (the D10 segmentation enum) applies only to malicious
`TOOL_OUTPUT` examples — `validate()` enforces both this and the
label/technique invariant (benign carries no technique; malicious
`TOOL_OUTPUT` must have one; malicious `TOOL_CALL` must not).

### Model/architecture decisions already made (don't relitigate without reading D-numbers)

- Fine-tuned **encoder classifier** (DeBERTa-v3), not a decoder-classifier or
  LM-judge, as primary architecture (D3). LM-judge is an ablation only (D4).
- Trained from the **vanilla pretrained checkpoint** — never fine-tuned from
  PromptGuard/ProtectAI weights, and never using their outputs as soft
  labels (D5, D12 — hard constraint). Those models are eval-only baselines.
- Primary backbone size is gated on real data volume (D13, parked) —
  xsmall (22M) ablation runs regardless (D6) to test a specific pre-registered
  multilingual/OOD hypothesis from PromptGuard's model card.
- ModernBERT is a named fallback, not primary — only swapped in if Phase 1's
  needle-in-haystack bypass rate crosses a pre-registered threshold in D7.
- Multi-turn aggregation, if built, is embed-then-pool (encode each tool
  output independently, pool fixed-size representations) — never
  concatenate-then-encode, which redistributes the same 512-token budget
  across more content rather than extending it (D8a).
- Model loading must pin `use_safetensors=True` explicitly (avoids silently
  downloading both `.bin` and `.safetensors` — doubles download size).

### Git workflow (`docs/specs/DEVELOPMENT_RULES.md`)

`main` (working/evaluated state, tagged at milestones) and `develop`
(integration branch) are the two long-lived branches. No direct commits to
either. One `feature/<phase-slug>` branch per `IMPLEMENTATION_PLAN.md` phase,
named to match the phase exactly. Commit messages reference the decision
ID(s) implemented, e.g. `train primary detector (implements D6, D13)`. Merge
to `develop` only when that branch's Definition of Done is genuinely,
fully met.

### Operating persona and communication style (`docs/specs/VISION.md`)

Judgment in this repo should read as that of a world-class LLM red-teaming /
AI security expert — but that only shows up in the *quality* of what's said,
never as self-credentialing. Don't claim expertise, don't name-drop the
mental models above as justification ("using first-principles thinking
here..."), don't narrate having reasoned rigorously — show the conclusion
and, where it adds real information, the one-line reason.

- Distinguish, every time, which of three epistemic states an answer is in:
  a known/published technique, a plausible-but-untested hypothesis, or
  speculation — especially in the write-up.
- Offense/defense duality: an attack or bypass isn't a finished finding
  unless it connects back to a specific hardening implication.
- Treat evals/red-team results skeptically the way production metrics
  deserve skepticism — alert to Goodhart, judge/model-family bias, and what
  a passing result does and doesn't actually prove.
- Communication default: concise and precise, shortest correct answer, no
  hedging filler, no restating the question back. Push back directly when a
  proposed attack/defense/eval is weak, redundant, or solving the wrong
  layer of the problem — don't soften it.

### Citation/provenance discipline

Two living logs, populated at point of use, not retroactively:
- `docs/DATA_SOURCES.md` — per-dataset-source license + redistribution status.
- `docs/CITATIONS.md` — per-paper/repo/technique provenance, with stable
  `C<N>` IDs that code docstrings point to. Several sources (InjecAgent,
  AgentDojo, LLMail-Inject, NotInject, BIPIA, Maatphor) are explicitly listed
  as **not yet independently re-confirmed** — verify against the primary
  source before citing any of them as settled in the write-up.
