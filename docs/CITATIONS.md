# Citation Log — Code & Method Provenance

Tracks papers, repos, and techniques that implementation code is *based on*
— separate from `DATA_SOURCES.md`, which tracks dataset provenance. Populate
at the point a source is actually used, not retroactively (per
`specs/DEVELOPMENT_RULES.md`). Each entry gets a stable ID (`C1`, `C2`, ...)
that code docstrings can point to — don't renumber existing entries.

---

## Template (copy per new entry)

```
### C<N> — <short name>

- **Source**: <paper/repo name, link>
- **Taken**: <specific thing used — algorithm, architecture idea,
  hyperparameter choice, code pattern>
- **Independent work**: <what was changed, adapted, or built ourselves on top>
- **Used in**: <file path(s)>
- **License note** (if code, not just method, is reused): <license, and
  whether verbatim reuse or reimplementation from the described method>
```

---

## Entries

### C1 — Meta PromptGuard / PromptGuard 2 model cards (primary source, verified)

- **Source**: Meta, PurpleLlama repo, model cards:
  https://github.com/meta-llama/PurpleLlama/blob/main/Prompt-Guard/MODEL_CARD.md
  and https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md
- **Taken**: (1) verbatim statement that DeBERTa-xsmall has no multilingual
  pretraining, creating a documented performance gap between the 22M and
  86M PromptGuard 2 variants on multilingual data — motivates D6's
  pre-registered xsmall ablation. (2) PromptGuard's stated 512-token context
  window and its own recommendation to chunk-and-scan-in-parallel for longer
  inputs — motivates D7's ModernBERT gate premise. (3) PromptGuard 2 86M's
  reported 97.5% Recall @ 1% FPR — motivates D6's choice of eval metric for
  direct comparability.
- **Independent work**: none of these are reused as weights/data (see D5,
  D12 — training lineage is fully separate); used only as cited facts
  motivating our own architecture/eval decisions, and as an eval-only
  baseline we benchmark against, never train on or from.
- **Used in**: `docs/DECISIONS.md` D6, D7.
- **Verification status**: fetched and quoted directly from the primary
  source (not a secondhand paraphrase) on the date this project's docs were
  audited. Re-check before final write-up submission in case the model card
  is revised.

### C2 — Zhan, Fang, Panchal, Kang (2025), "Adaptive Attacks Break Defenses
     Against Indirect Prompt Injection Attacks on LLM Agents"

- **Source**: NAACL 2025 Findings, pp. 7116-7132. arXiv:2503.00061.
  https://aclanthology.org/2025.findings-naacl.395/
- **Taken**: motivating citation for D2 (why IPI has a robustness literature
  worth building red-team methodology against) and D10 (the general finding
  that published defenses evaluated only against static attacks show
  inflated robustness — over 50% ASR once adaptively attacked).
- **Independent work**: our red-team harness (D10) is our own design, not a
  reimplementation of this paper's specific GCG/AutoDAN-based adaptive
  attack methods — it borrows the *principle* (test adaptively, not just
  statically) rather than the specific optimizer.
- **Used in**: `docs/DECISIONS.md` D2, D10.
- **Verification status**: confirmed via multiple independent sources
  including ACL Anthology and arXiv abstract page. Correctly cited.

### C3 — Hackett, Birch, Trawicki, Suri, Garraghan (2025), "Bypassing LLM
     Guardrails: An Empirical Analysis of Evasion Attacks against Prompt
     Injection and Jailbreak Detection Systems"

- **Source**: LLMSEC 2025 workshop (ACL Anthology), arXiv:2504.11168.
  https://aclanthology.org/2025.llmsec-1.8/
- **Taken**: concrete, citable technique list for D10's manual seed attack
  categories — character injection (homoglyphs, zero-width chars, spacing,
  full-width chars) and adversarial-ML evasion, empirically tested against
  Meta PromptGuard, ProtectAI, NeMo Guard, and Vijil specifically — the same
  class of guardrail we're building and benchmarking against.
- **Independent work**: our seed attacks are authored/adapted by us (per the
  problem statement's non-100%-LLM-generated requirement), using this
  paper's technique taxonomy as a reference list, not its literal test
  prompts.
- **Used in**: `docs/DECISIONS.md` D10.
- **Verification status**: confirmed, correct title/authors/venue/arXiv ID.

### C4 — Fairoze, Garg, Lee, Wang (2025), "Bypassing Prompt Guards in
     Production with Controlled-Release Prompting"

- **Source**: arXiv:2510.01529 (UC Berkeley). https://arxiv.org/abs/2510.01529
- **Taken**: the "spaced-release" attack variant specifically exploits a
  guard model's limited context window versus the protected model's larger
  one — direct motivation for D10's needle-in-haystack / long-carrier-
  document seed category, and for D7's ModernBERT gate premise.
- **Independent work**: our needle-in-haystack tests are our own
  constructions targeting DeBERTa's specific 512-token limit, not this
  paper's cipher-based timed-release encoding scheme (which targets
  full jailbreak prompts on production chat platforms, a different setting
  than our tool-output classifier).
- **Used in**: `docs/DECISIONS.md` D7, D10.
- **Verification status**: confirmed, correct title/authors/arXiv ID.

---

## Verification status — resolved

All items originally listed here (InjecAgent, AgentDojo, LLMail-Inject,
NotInject, BIPIA) have since been individually fetched and verified
directly — see C6-C10 below. Only Maatphor (LLM-guided attack mutation
methodology referenced in D10) remains genuinely unverified — it was never
independently re-checked in this conversation; confirm before citing it in
the final write-up.

**Remaining pending list (found later, deliberately not pursued further
per D20 — infrastructure cost vs. already-adequate sources):** TrojanTools,
Agent Security Bench (ASB), MCP-SafetyBench — all confirmed real (see C11)
but excluded on cost/category grounds, not citation-uncertainty grounds.
Two further leads, genuinely unverified: `Necent/llm-jailbreak-prompt-
injection-dataset` (HF aggregator) and MCP-AttackBench — optional Phase 0
checks, not required (D20).

### C5 — Greshake, Abdelnabi, Mishra, Endres, Holz, Fritz (2023), "Not What
     You've Signed Up For: Compromising Real-World LLM-Integrated
     Applications with Indirect Prompt Injection"

- **Source**: AISec'23 (ACM), arXiv:2302.12173.
  https://arxiv.org/abs/2302.12173
- **Taken**: this is the foundational paper that coined "indirect prompt
  injection" and established the taxonomy (data theft, worming, ecosystem
  contamination, unauthorized API calls) motivating the entire threat model
  behind D2's surface choice. Not a supporting citation — the definitional one.
- **Independent work**: our schema, dataset, and detector are our own
  construction; this paper is cited for the concept definition, not reused
  as data or code.
- **Used in**: `docs/DECISIONS.md` D2.
- **Verification status**: confirmed via arXiv, ACM DL, and independent
  corroboration across multiple secondary sources (OWASP LLM01, Microsoft
  MSRC's own indirect-injection defense writeup, Wikipedia's prompt
  injection article). Correctly cited.

### C6 — Real, concretely available public datasets for the IPI surface
     (found during dataset-planning discussion, D14; license/access fully
     verified in D19)

- **`MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT`** (HuggingFace):
  35,000 malicious / 35,000 benign pulled, derived directly from the actual
  BIPIA benchmark (Yi et al.), schema `{"context", "user_intent", "label",
  "source"}` — maps closely onto our own schema. **CC BY-SA 4.0 claimed
  (blanket), confirmed directly from the dataset card. Gated — requires HF
  account + terms agreement**, confirmed by direct fetch.
  **License-provenance finding, not a formality**: ~48.3%
  (16,919/35,000) of this dataset's *malicious* rows are content-pattern-
  matched to BIPIA's WebQA/Summarization domains (XSum, NewsQA) — content
  Microsoft's own `microsoft/BIPIA` README explicitly excludes from
  redistribution "due to license issues." The blanket CC BY-SA 4.0 claim
  almost certainly does not correctly cover that content. **Consequence:
  this project uses only MAlmasabi's benign half (independently
  GPT-4o-mini-synthetic, not implicated in the same provenance issue) — the
  malicious half is excluded from training entirely, not merely capped.**
  Final contribution to the committed dataset: **2,521 benign rows** (a
  coherence-filtered, domain-stratified sample of the pulled 35,000 benign
  rows), 0 malicious.
  https://huggingface.co/datasets/MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT
- **`leolee99/NotInject`** (HuggingFace, from the InjecGuard/ACL 2025 paper,
  arXiv:2410.22770): 0 malicious / 339 benign (entirely benign by design —
  not a mixed set), schema
  `{prompt, word_list, category}`. **MIT license, confirmed directly. Not
  gated** — loads immediately, no account needed. Final contribution:
  **all 339 rows used unchanged.**
  https://huggingface.co/datasets/leolee99/NotInject
- **`prodnull/prompt-injection-repo-dataset`** (HuggingFace, part of
  "CloneGuard"): 2,916 malicious / 2,755 benign, purpose-built for
  coding-agent tool-output injection specifically — the closest existing
  sibling project to this one. **Apache 2.0, confirmed directly. Gated**
  (HF account + terms agreement). Includes documented hard negatives
  (security docs, CVEs, docstrings) and a reported real OOD-generalization
  failure (43% FPR against MCP tool-result samples from a different
  distribution) directly relevant to D6/D7/D20's generalization concerns.
  Final contribution: **all 5,671 rows used unchanged** (2,916 malicious /
  2,755 benign) — every malicious row assigned `InjectionTechnique.OTHER`,
  since the card's own 24-category taxonomy isn't a column in this release.
  https://huggingface.co/datasets/prodnull/prompt-injection-repo-dataset
- **Final composition, for the record (see `LICENSE-THIRD-PARTY.md` at the
  repo root for the full per-license breakdown and row-count
  reconciliation — this file and that one are the public source of truth
  for dataset provenance)**: of `data/processed/curated_pool.jsonl`'s
  13,032 rows, these three sources contribute 5,671 (prodnull) + 2,521
  (MAlmasabi, benign only) + 339 (NotInject) = 8,531; the remaining 4,501
  come from BIPIA directly (C7).
- **Independent work**: none of these are training-lineage reuse (D5/D12
  unaffected — these are data sources, not weights or soft-labels); used as
  training/eval data per D2's public-source strategy.
- **Used in**: `docs/DECISIONS.md` D14, D19; `LICENSE-THIRD-PARTY.md`.
- **Verification status**: fully confirmed by direct fetch of each dataset
  card (not secondhand). No longer pending.

### C7 — `microsoft/BIPIA` (original repo), Yi, Xie, Zhu, Hines, Kiciman,
     Sun, Xie, Wu (2023), "Benchmarking and Defending Against Indirect
     Prompt Injection Attacks on Large Language Models"

- **Source**: `github.com/microsoft/BIPIA`, arXiv:2312.14197, KDD 2025.
- **Taken**: originally planned to reuse `AutoPIABuilder`/`QAPIABuilder`
  wholesale (as this entry said pre-implementation); actual implementation
  (`src/data/sources/bipia.py`) reuses only `bipia.data.utils.insert_end` /
  `insert_start` / `insert_middle` — the sentence-boundary-aware insertion
  primitives (nltk `PunktSentenceTokenizer`-based for `insert_middle`) — and
  writes its own combination/pairing loop instead of going through
  `construct_samples()`. Reason (logged in `bipia.py`'s own docstring, not
  just here): the builder classes are built for constructing benchmark chat
  prompts (system/user templates for evaluating an LLM under test), a
  different job than ours (matched clean/poisoned content pairs for a
  classifier dataset) — and relying on `construct_samples()`'s returned row
  order to recover which poisoned row came from which clean context would
  depend on an undocumented internal iteration-order guarantee. Five task
  domains (EmailQA, WebQA, Summarization, TableQA, CodeQA) exist in the
  original benchmark; only EmailQA, TableQA, CodeQA are directly bundled —
  WebQA and Summarization require fetching from their original sources due
  to license terms, per Microsoft's own README. The same clean context rows,
  used without insertion, are our matched-pair benign class (D19).
- **License, verified by reading the actual LICENSE file (not assumed)**:
  code MIT (Microsoft Corporation); WikiTableQuestions (TableQA) CC BY-SA
  4.0; Stack Exchange (CodeQA) CC BY-SA 4.0; OpenAI Evals invoices
  (EmailQA) MIT. Mixed, not the single blanket license MAlmasabi's
  derivative implies (C6) — MAlmasabi's blanket "CC BY-SA 4.0" claim is
  imprecise relative to this, the actual primary source.
- **Independent work**: the combination/pairing loop and schema mapping in
  `src/data/sources/bipia.py` are ours; only the insertion primitives
  (`insert_end`/`insert_start`/`insert_middle`) and the content corpora are
  theirs, per the license terms above.
- **Final composition contributed to `data/processed/curated_pool.jsonl`**:
  **4,501 rows** (3,600 malicious, stratified-sampled by
  `(domain, template, insertion_position)` at cap=6/stratum, not BIPIA's
  full 41,250-row generative capacity; 901 benign, the full offer from
  BIPIA's own corpus under a corrected dedup rule), split by domain and
  therefore by which license actually governs each row: EmailQA
  (MIT) 1,434 rows, TableQA + CodeQA (CC BY-SA 4.0) 3,067 rows. Full
  per-domain breakdown in `LICENSE-THIRD-PARTY.md`.
- **Used in**: `docs/DECISIONS.md` D19, D22; `src/data/sources/bipia.py`;
  `LICENSE-THIRD-PARTY.md`.
- **Verification status**: fully confirmed — repo README and LICENSE file
  both fetched and read directly.

### C8 — Zhan, Liang, Ying, Kang (2024), "InjecAgent: Benchmarking
     Indirect Prompt Injections in Tool-Integrated Large Language Model
     Agents"

- **Source**: ACL Findings 2024, arXiv:2403.02691,
  `github.com/uiuc-kang-lab/InjecAgent`.
- **Taken**: cited as a real, verified benchmark for context (D2's
  literature landscape) — 1,054 test cases, 17 user tools, 62 attacker
  tools, grounded in Ruan et al. 2023's 330 real tool definitions across 36
  toolkits. GPT-4-generated (+ manual refinement) tool-response templates
  with an `<Attacker Instruction>` placeholder.
- **Not used as training data** — see D19: no benign class exists in this
  dataset (attack templates only), so it doesn't fit our schema's paired
  structure without an unrelated, non-matched benign source.
- **Used in**: `docs/DECISIONS.md` D19.
- **Verification status**: fully confirmed — GitHub repo and paper
  abstract both fetched directly. License not stated in what was fetched;
  not used, so not blocking.

### C9 — Debenedetti, Zhang, Balunović, Beurer-Kellner, Fischer, Tramèr
     (2024), "AgentDojo: A Dynamic Environment to Evaluate Prompt
     Injection Attacks and Defenses for LLM Agents"

- **Source**: NeurIPS 2024, arXiv:2406.13352,
  `github.com/ethz-spylab/agentdojo`. **MIT license, confirmed directly
  from the paper's own text** ("released...under MIT license").
- **Taken**: cited as a real, verified live evaluation framework — 97 user
  tasks, four domains (banking, workspace, Slack, travel), ~629-949
  (user-task, injection-task) pairs depending on version, multiple named
  attack templates (`ignore_previous`, `system_msg`, `important_inst`).
- **Not used as training data** — it is a Python package you run against a
  live simulated environment, not a static file (D19, D20). A third-party
  paper (Agent-Sentry Bench) independently criticizes it as having too few
  hand-written tasks to train a structural classifier — external
  corroboration, not just our own assessment.
- **Used in**: `docs/DECISIONS.md` D19, D20.
- **Verification status**: fully confirmed — GitHub repo, paper text, and
  license statement all fetched directly.

### C10 — Abdelnabi et al. (2025), "LLMail-Inject: A Dataset from a
     Realistic Adaptive Prompt Injection Challenge"

- **Source**: arXiv:2506.09956.
- **Taken**: cited as a real, large-scale adversarial submission dataset —
  208,095 unique prompts (169,598 phase 1, 38,497 phase 2) from a live
  red-teaming competition against an email-assistant agent, with a
  ground-truth-annotated subset of 25,323 submissions confirmed to have
  triggered the target tool.
- **Not used as training data** — per the correction in D19: the split is
  not clean binary (25,323 confirmed tool-triggering vs. ~182,772
  *unconfirmed*, not confirmed-benign), and label quality depends on an
  "LLM-annotator" judgment call rather than deterministic ground truth.
  Its participant count (839, cited in some secondary sources) was never
  independently confirmed by us — flagging explicitly, since the
  submission counts above ARE self-verified (fetched from the paper's own
  appendix) and the two shouldn't be treated at the same confidence level.
- **Used in**: `docs/DECISIONS.md` D19.
- **Verification status**: submission/annotation counts confirmed by
  direct fetch of the paper appendix. Participant count NOT independently
  confirmed — secondhand only.

### C12 — Ash & Adams (2020), "On Warm-Starting Neural Network Training",
     plus supporting context on adversarial-harvest retraining practice

- **Source**: Ash & Adams, NeurIPS 2020,
  https://proceedings.neurips.cc/paper/2020/file/288cd2567953f06e460a33951f55daaf-Paper.pdf.
  Supporting context (industry-practice framing, not load-bearing claims):
  "Automated Adversarial Discovery for Safety Classifiers"
  (https://arxiv.org/html/2406.17104v1), "Robust Safety Classifier for
  Large Language Models: Adversarial Prompt Shield"
  (https://arxiv.org/pdf/2311.00172), "Adversarial Training & Secure
  Fine-Tuning LLMs"
  (https://apxml.com/courses/intro-llm-red-teaming/chapter-5-defenses-mitigation-strategies-llms/adversarial-training-security-fine-tuning).
- **Taken**: Ash & Adams' core finding — warm-starting on an
  incrementally-grown dataset tends to generalize worse than training from
  scratch on the combined dataset, even though it converges faster —
  motivated the harvest-retrain methodology discussion (whether to
  continue-train from `models/primary/epoch_3` or retrain from the vanilla
  backbone on curated+self_authored+harvested). Also used to correctly
  frame that large-model/LLM safety-tuning's common preference for
  continued fine-tuning is a **compute-cost** convention, not evidence that
  warm-starting generalizes better — a distinction that matters at this
  project's model scale (142M params, full retrain measured in hours, not
  weeks) where the compute-cost argument doesn't hold the way it does for
  much larger models.
- **Independent work**: the final methodology call (see
  `docs/DECISIONS.md` — logged alongside the harvest-retrain decision) is
  our own, weighing this generalization-risk finding against this
  project's specific constraints (harvest size relative to pool size, time
  budget, and the stated goal of the retrain) — not a mechanical
  application of the paper's recommendation.
- **Used in**: harvest-retrain methodology decision (`docs/DECISIONS.md`,
  Phase 1).
- **Verification status**: Ash & Adams fetched directly (arXiv/NeurIPS
  proceedings PDF, primary source). The three supporting links are search
  summaries, not independently fetched and read in full — treat as
  contextual framing, not independently verified claims, until re-checked.

### C11 — TrojanTools, Agent Security Bench (ASB), MCP-SafetyBench
     (investigated, confirmed real, excluded per D20)

- **TrojanTools**: OpenReview 2025, "TrojanTools: Adaptive Indirect Prompt
  Injection on LLM Agents via Malicious Tool-Calling." An attack-generation
  *framework*, not a dataset — its internal dataset ("IPAF") has no
  confirmed download link found.
- **Agent Security Bench (ASB)**: Zhang et al., ICLR 2025, arXiv:2410.02644,
  `github.com/agiresearch/ASB`. 10 scenarios, 400+ tools, 27 attack/defense
  methods. A third-party benchmark index explicitly classifies it as
  "Harness," distinct from entries it labels "Dataset."
- **MCP-SafetyBench**: Zong, Shen, Wang, Lan, Yang (2025), arXiv:2512.15163,
  `github.com/xjzzzzzzzz/MCPSafety`. Real MCP servers, 5 domains including
  "repository management," 20 attack types. Requires Docker, live MCP
  server instances, API keys, and an explicitly-recommended disposable
  GitHub account (performs real repository operations).
- **Not used**: all three are live evaluation harnesses/frameworks, not
  flat training data, and MCP-SafetyBench specifically crosses from
  "compute" into "infrastructure" cost (D20) relative to already-adequate
  existing sources.
- **Used in**: `docs/DECISIONS.md` D20.
- **Verification status**: all three fully confirmed real via direct
  fetch (GitHub repos, arXiv abstracts, or OpenReview page). Excluded on
  cost/category grounds, not citation-uncertainty grounds.
- **Used in**: `docs/DECISIONS.md` D14, `docs/DATA_SOURCES.md`.
- **Verification status**: dataset existence, scale, and schema confirmed
  directly from HuggingFace dataset pages. Exact license field for the
  latter two needs direct confirmation on the HF page itself in Phase 0
  before use (the search snippet didn't surface it explicitly).
