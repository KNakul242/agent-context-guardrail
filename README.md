# Agent Context Guardrail — Tool-Output Indirect Prompt Injection Detector

RepelloAI Research Engineer Assignment — Track 1 (Agentic Tool-Call Guardrail).

Scope: detecting **indirect prompt injection (IPI)** in tool outputs consumed by
LLM agents (files, webpages, API responses) that an agent reads mid-task.
Malicious/dangerous tool-call detection is a stretch extension — see `docs/DECISIONS.md`.

## Why this scope, this architecture, this everything

Every non-obvious choice in this repo — attack surface, backbone, why we didn't
fine-tune from PromptGuard's checkpoint, why encoder over decoder-classifier over
LM-judge, the ModernBERT fallback trigger condition, the multi-turn aggregation
design — is logged with reasoning in **`docs/DECISIONS.md`**. That file is the
primary artifact for understanding *how* we got here; this README is just setup.

## Repo structure

```
data/
  raw/            # untouched pulls from public sources (InjecAgent, NotInject, LLMail-Inject, ...)
  processed/      # cleaned, labeled, split train/val/test — our unified schema
  redteam/        # adversarial examples we construct/harvest to break our own detector
src/
  data/           # dataset construction, cleaning, schema, license tracking
  model/          # encoder classifier: train.py, model definition, aggregation module
  eval/           # metrics: F1/AUC, recall@low-FPR, FPR on hard-negatives, red-team ASR
  redteam/        # attack generation: manual seeds + LLM-red-teamer escalation loop
notebooks/        # exploratory work only — nothing load-bearing lives only in a notebook
models/           # trained checkpoints / adapters (also mirrored to HF Hub, see below)
docs/
  DECISIONS.md    # running design-decision log with rationale (see above)
  DATA_SOURCES.md # per-source license + attribution tracking
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two execution environments, used deliberately for different steps:
- **Local (Mac, MPS backend)** — all classifier training/eval. DeBERTa-v3-small
  trains in minutes on MPS; no need to burn Colab GPU-hours on this.
- **Colab (free tier)** — reserved for calling a stronger free-tier model
  (OpenRouter) to assist synthetic data generation and to run the LM-judge
  baseline comparison, per `docs/DECISIONS.md`.

## Run order

1. `src/data/` — build the labeled dataset (see `docs/DATA_SOURCES.md` for source list)
2. `src/model/train.py` — train the primary detector (DeBERTa-v3-small)
3. `src/redteam/` — run seed attacks + LLM-red-teamer escalation loop against the trained model
4. `src/model/train.py --resume` — retrain on harvested red-team bypasses
5. `src/eval/` — full quantitative eval: in-distribution metrics, hard-negative FPR, red-team ASR by category

## Artifacts

Trained weights and datasets are published on HuggingFace Hub at: `<TBD — filled in before submission>`

## Citations

See `docs/DECISIONS.md` and `docs/DATA_SOURCES.md` for full citations of every
paper, dataset, and repo this work builds on, with a one-line note on what was
taken vs. what was implemented ourselves.
