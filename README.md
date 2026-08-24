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
  raw/            # untouched pulls from public sources (prodnull, NotInject, MAlmasabi, BIPIA — see docs/DATA_SOURCES.md)
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

`requirements.txt` is exact-pinned, not floor-pinned (`torch==2.13.0`
etc.) — a floor-only install (`torch>=2.2`) was the actual root cause of
a real training-collapse bug this project hit (`docs/ISSUES.md` ISSUE-3);
don't loosen the pins without reading that entry first. If you're on
Colab, do **not** add a `torchvision` pin either — see the comment above
that line in `requirements.txt` (ISSUE-7).

For the red-team escalation loop and the manual example-authoring tools
(`scripts/escalate_redteam.py`, `scripts/example_lab_gemini.py`,
`scripts/example_lab_redteam.py`), create `secrets.env` in the repo root
with `GEMINI_API_KEY=...` (Google AI Studio free tier).

Two execution environments, used for different steps:
- **Local (Mac, MPS backend)** — classifier training/eval. Correction to
  an earlier claim here: DeBERTa-v3-small does **not** train in minutes at
  real data scale (~10K rows) — a full 3-epoch run measured ~2-8 hours
  depending on dtype correctness (`docs/ISSUES.md` ISSUE-1/ISSUE-3).
- **Colab (free tier, T4)** — used for the LLM-red-teamer escalation
  calls (Gemini, not OpenRouter — see `src/redteam/llm_client.py`), and,
  under deadline pressure, also authorized as a parallel/redundant
  environment for the primary training run itself
  (`docs/DECISIONS.md` D29) via `notebooks/colab_train_primary.ipynb`.

## Dataset

`data/processed/curated_pool.jsonl` — **13,032 rows, exactly 6,516
malicious / 6,516 benign** — composed from four public sources plus a
growing self-authored slice. Full derivation (sampling/filtering
parameters, per-domain breakdowns, the license-provenance finding that
drove excluding one source's malicious half entirely) is in
`docs/data_summary.md`; this is the summary.

| Source | Rows | Malicious | Benign | License |
|---|---|---|---|---|
| `prodnull/prompt-injection-repo-dataset` | 5,671 | 2,916 | 2,755 | Apache 2.0 |
| `microsoft/BIPIA` (own insertion primitives, own pairing loop) | 4,501 | 3,600 | 901 | Mixed — MIT (EmailQA) / CC BY-SA 4.0 (TableQA, CodeQA) |
| `MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT` (benign half only — malicious half excluded, license-provenance gap, see `LICENSE-THIRD-PARTY.md`) | 2,521 | 0 | 2,521 | CC BY-SA 4.0 |
| `leolee99/NotInject` (hard-negative supplement) | 339 | 0 | 339 | MIT |
| **Curated subtotal** | **13,032** | **6,516** | **6,516** | mixed — see `LICENSE-THIRD-PARTY.md` |
| Self-authored (`self_authored.jsonl`, Track 0B, hand-authored/adapted, growing) | 3 | 1 | 2 | this repo's `LICENSE` |

**Split**: `data/processed/{train,val,test}.jsonl` — 10,430 / 1,293 / 1,309
rows, document-group-aware stratified split (`docs/DECISIONS.md` D25) so a
BIPIA document's clean and poisoned forms always land in the same split —
verified zero leaks across 901 document groups. Self-authored rows merge
in at training time (`curated + self_authored`), not written into the
split files themselves.

**Red-team probe corpus**: `data/redteam/seeds.jsonl` — 35 hand-authored/
adapted seeds across all 9 `InjectionTechnique` categories (D10's
taxonomy), separate from training data by design — only *confirmed
bypasses* get harvested into training-shaped examples
(`data/redteam/harvested*.jsonl`, see Results below).

License obligations (what requires attribution, what requires
share-alike, exact per-source row counts) are in `LICENSE-THIRD-PARTY.md`
— **`data/processed/` is mixed-license, not covered by this repo's own
`LICENSE`.**

## Run order

Real entrypoints live in `scripts/`, not as `__main__` blocks inside
`src/` (each `src/` module is imported by its corresponding script and
by `tests/`, not run directly).

1. **Build the dataset**: `python3 scripts/build_curated_pool.py` (add
   `--pull` to also re-fetch each public source first — needs `hf auth
   login` for the gated sources; see `docs/DATA_SOURCES.md`), then
   `python3 scripts/split_curated_pool.py` to materialize
   `data/processed/{train,val,test}.jsonl`.
2. **Train the primary detector**:
   ```
   python3 scripts/train_primary.py --success-criterion "..." \
       [--epochs N] [--batch-size N] [--lr F] [--seed N] \
       [--limit N] [--backbone HF_ID_OR_LOCAL_DIR] [--output-dir DIR]
   ```
   `--limit N` runs a cheap smoke test against real data/model before
   committing to a full run — use it for any change to the training path
   (see ISSUE-1's postmortem for why). `--backbone` defaults to the
   vanilla `microsoft/deberta-v3-small`; pass a local checkpoint dir
   (e.g. `models/primary/epoch_3`) to warm-start instead (see D31 for
   when that's actually the right call vs. training from vanilla).
   The reported primary checkpoint (`models/primary/epoch_3`) was trained
   with `--epochs 3 --batch-size 16 --lr 2e-5 --seed 0` (vanilla backbone).
3. **Evaluate**: `python3 scripts/evaluate.py --checkpoint
   models/primary/epoch_3 --split data/processed/val.jsonl` — prints
   F1/ROC-AUC/recall@1%-FPR/hard-negative-FPR.
4. **Red-team**: `python3 scripts/run_redteam.py --checkpoint
   models/primary/epoch_3 --derive-threshold-from-val
   data/processed/val.jsonl --max-fpr 0.01` (bypass rate by
   `InjectionTechnique`, never one aggregate number — D10). Then
   `python3 scripts/escalate_redteam.py --checkpoint models/primary/epoch_3
   [--max-rounds N]` for the automated LLM-mutation escalation loop
   (needs `GEMINI_API_KEY`, see Setup).
5. **Harvest-retrain** (optional, only worth it if the harvest is large
   enough to matter relative to the pool — see D31 for how that call was
   actually made): merge confirmed bypasses from `data/redteam/
   harvested*.jsonl` into training data, then re-run step 2 with
   `--backbone` pointed at the checkpoint you're continuing from.

## Results

Full derivation, investigation trails, and honest corrections for every
number below live in `docs/ISSUES.md` and `docs/DECISIONS.md` — this is a
summary table, not a substitute for reading those if you want the *why*.

### Primary model — in-distribution metrics

Trained from vanilla `microsoft/deberta-v3-small`, 3 epochs, seeded
(`seed=0`), fp32-pinned, 10,433 rows (10,430 curated + 3 self-authored).
Run independently on two different hardware/software stacks as a
cross-check that the training pipeline (not just one environment's
numerics) is correct:

| | Local (MPS), `epoch_3` | Colab (CUDA), `epoch_3` |
|---|---|---|
| F1 | 0.968 | 0.972 |
| ROC-AUC | 0.995 | 0.997 |
| recall@1%-FPR | 0.950 | 0.953 |
| hard-negative FPR | 0.088 (3/34) | 0.029 (1/34) |

The two independent runs land within a few thousandths of each other on
every metric — strong convergent evidence the pipeline (and specifically
the fp32 dtype fix, `docs/ISSUES.md` ISSUE-3) is correct and
hardware-independent, not an artifact of one environment.

### Red-team — base pass

Seed corpus run once (no mutation) against the local `epoch_3` checkpoint,
at the threshold that actually achieves 1%-FPR on `val.jsonl` (0.7413 —
deriving red-team results at the arbitrary default 0.5 would answer a
different question than the eval report's headline metric):

| Technique | Bypass rate |
|---|---|
| direct_override | 0.0% |
| fake_system_tag | 0.0% |
| role_reframe | 0.0% |
| encoding_obfuscation | 0.0% |
| unicode_obfuscation | 0.0% |
| payload_split | 0.0% |
| fictional_framing | 0.0% |
| needle_in_haystack | see below — blended number is misleading, decomposed |
| low_resource_language | 62.5% (5/8) |

**`needle_in_haystack` needed correction, not just measurement**
(`docs/ISSUES.md` ISSUE-9): the original 3 seeds placed their injected
instruction near the document midpoint, past the model's 512-token
truncation window for the two longest seeds — the model never saw the
payload for those, so the raw bypass rate was measuring truncation, not
detection. Fixed by splitting the metric: **in-window (payload survived
truncation) bypass rate = 0.0%** — zero evidence of a real semantic
weakness; **out-of-window bypass rate = 100%**, but that's tautological
(no signal reaches the classifier on those inputs) and is, if anything,
evidence for a longer-context architecture rather than a training
problem. Full methodology in ISSUE-9.

**`low_resource_language` is a real, substantive weakness**, not an
artifact — held at 62.5% (5/8) even after adding 5 new, genuinely
native-authored seeds (Slovenian, Basque, Javanese confirmed bypassing;
Somali, Welsh caught) specifically to rule out the original 3 seeds'
machine-translation quality as the explanation (`docs/DECISIONS.md` D32).

### Red-team — automated escalation loop (LLM-mutation, D10)

A static single-shot probe significantly overstates robustness on most
techniques. Each seed the base pass caught was mutated (Gemini,
Maatphor-style) up to 5 rounds against the live classifier until it
bypassed or the round budget ran out:

| Technique | Base pass | After escalation |
|---|---|---|
| direct_override | 0% | 0/3 — robust |
| fake_system_tag | 0% | 0/3 — robust |
| role_reframe | 0% | 2/3 bypassed |
| encoding_obfuscation | 0% | 1/3 bypassed |
| unicode_obfuscation | 0% | **3/3 — fully compromised** |
| payload_split | 0% | 2/3 bypassed |
| fictional_framing | 0% | 2/3 bypassed |
| needle_in_haystack (in-window) | 0% | 0/3 — robust, unaffected |
| needle_in_haystack (out-of-window) | 100%* | 3/3* — unaffected (truncation artifact, ISSUE-9) |
| low_resource_language | 66.7% (original 3-seed base rate at time of escalation) | 2/3 immediate + 1/3 after full escalation |

**5 of 7 techniques that looked fully robust in the static pass collapsed
under real mutation.** This is the actual headline red-team finding —
the base pass alone would have significantly overstated this detector's
robustness. 15 confirmed bypasses harvested (base + escalation combined,
`data/redteam/harvested_merged.jsonl`).

### Harvest-retrain — result, reported honestly including what's unresolved

Warm-started from `epoch_3` (not retrained from vanilla — see
`docs/DECISIONS.md` D31 for the reasoning, including why that call was
initially made the other way and then corrected), 2 epochs, on
curated + self_authored + the 15-example harvest, run on Colab given the
data volume relative to the ~10,433-row pool (~0.1%) didn't justify the
time cost of a full retrain locally:

| | Base (`epoch_3`) | Retrained (`epoch_2` warm-start) |
|---|---|---|
| F1 | 0.968 | 0.961 |
| ROC-AUC | 0.995 | 0.992 |
| recall@1%-FPR | 0.950 | 0.936 |
| hard-negative FPR | 0.088 (3/34) | 0.088 (3/34) |

Small regression on the standard metrics — within ordinary fine-tuning
noise at this data scale, not a clear signal either way.

- **`low_resource_language`: confirmed improvement** — 62.5% → 0% bypass,
  same 8 seeds, same static method both times, a clean and valid
  comparison.
- **`needle_in_haystack`: confirmed unaffected**, exactly as predicted —
  architectural, not something training data fixes.
- **The 5 escalation-collapsing techniques: genuinely unresolved, not
  "fixed."** The retrained model's static-pass numbers on these look
  identical to the *original* model's static-pass numbers before
  escalation ever ran (both 0%) — that comparison doesn't test what
  matters, since the weakness was only ever exposed by mutation. Properly
  answering "did the retrain help" here requires re-running escalation
  against the retrained checkpoint, which was **not done, under explicit
  time constraints** (`docs/DECISIONS.md` D31). Stated as a known
  limitation, not glossed over as either a success or a failure.

## Artifacts

Per `docs/DECISIONS.md` D33: **datasets are committed directly in this
repo** (`data/processed/`, `data/redteam/`) — see `LICENSE-THIRD-PARTY.md`
for per-source license/attribution obligations (the dataset folder is
mixed-license, not covered by this repo's own `LICENSE`). **Trained model
weights are published on HuggingFace Hub**:

- **Primary/submitted model**: https://huggingface.co/knakul242/agent-context-guardrail-primary
  (DeBERTa-v3-small, `epoch_3`, trained from vanilla, the results reported
  above under "Local (MPS)")
- **Experimental harvest-retrain checkpoint**: https://huggingface.co/knakul242/agent-context-guardrail-primary-retrain-experimental
  (warm-started from the primary checkpoint, 2 epochs on 15 harvested
  red-team bypasses — explicitly **not** the primary model; its own model
  card states the small eval regression and the unresolved
  re-escalation-test limitation, D31, rather than glossing over either)

Data commit: `8e898ab` on `feature/primary-detector-redteam`.

## License

Code and self-authored data: see `LICENSE`. Third-party dataset rows
committed into `data/processed/` carry their own upstream licenses
(Apache 2.0, MIT, and CC BY-SA 4.0 — the latter under share-alike
obligations that apply to specific rows regardless of this repo's own
license) — see `LICENSE-THIRD-PARTY.md` for the full per-source
breakdown and exact row counts.

## Write-up

`REPORT.md` (repo root) has the full methodology, design decisions,
rejected alternatives, and known-weaknesses discussion required by the
assignment — this README is setup + results, not the write-up itself.
`<TBD — not yet written as of this pass>`.

## Citations

See `docs/DECISIONS.md`, `docs/CITATIONS.md`, and `docs/DATA_SOURCES.md`
for full citations of every paper, dataset, and repo this work builds on,
with a one-line note on what was taken vs. what was implemented ourselves.
