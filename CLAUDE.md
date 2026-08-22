# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A take-home assignment (`docs/specs/PROBLEM_STATEMENT.md`,
verbatim, ground truth). Track 1 (Agentic Tool-Call Guardrail): train a
detector for **indirect prompt injection (IPI)** hidden in tool outputs
(files, webpages, API responses) that an LLM agent reads mid-task.
Malicious/dangerous tool-call detection is an in-scope stretch extension,
built on the same schema, only after the IPI pipeline is done.

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

Only `src/data/schema.py` is implemented; the repo is at Phase 0 (Data
Foundation, `docs/specs/IMPLEMENTATION_PLAN.md`). `data/{raw,processed,redteam}/`
are empty (gitkept). No model/eval/redteam code exists yet. Phase 0 ends at a
**hard gate**: a data summary report must be reviewed and D13 (primary model
size, parked) resolved in `docs/DECISIONS.md` before any Phase 1 branch opens.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# MPS backend sanity check (Mac) — forward+backward pass, NaN/Inf grad check
python mps_smoke_test.py
```

No test suite, lint config, or build step exists yet — `docs/specs/DEVELOPMENT_RULES.md`
requires deterministic code (schema, pipeline transforms, aggregation/pooling
module, eval metrics, red-team mutation logic) to be written test-first, but
no test framework has been added to this repo yet.

Two execution environments, used deliberately:
- **Local (Mac, MPS backend)** — all classifier training/eval. DeBERTa-v3-small
  trains in minutes on MPS.
- **Colab (free tier)** — reserved for calling a stronger free-tier model
  (OpenRouter) for synthetic data generation and the LM-judge baseline.

## Architecture

```
data/
  raw/            untouched pulls from public sources (InjecAgent, AgentDojo,
                   LLMail-Inject, NotInject, BIPIA — see docs/DATA_SOURCES.md)
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

`InjectionTechnique` is the category enum every red-team/eval result gets
segmented by (D10) — results are reported per-category, never as one
aggregate ASR/bypass-rate number.

`validate(ex)` enforces the label/technique invariant: benign examples carry
no technique, malicious examples must.

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
