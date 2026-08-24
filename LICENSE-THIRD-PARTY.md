# Third-Party Data Licenses & Redistribution Obligations

Scope: every external data source whose rows are committed into
`data/processed/` in this repo. **This file covers data only** — code
licensing (this repo's own `LICENSE`) is separate.

Every license claim below was confirmed directly against the dataset's
own card/repo (see `docs/CITATIONS.md` for the paper/repo provenance
notes). Row counts are pulled from the actual committed output —
`data/processed/curated_pool.jsonl`, counted directly, per source — not
from earlier pull-time pool sizes, which describe what was *pulled*, not
what was *kept*. Where the two differ, that's noted explicitly below, not
silently reconciled.

---

## `prodnull/prompt-injection-repo-dataset`

- **Link**: https://huggingface.co/datasets/prodnull/prompt-injection-repo-dataset
- **License**: Apache 2.0 (confirmed directly from the dataset card).
- **Attribution notice required (Apache 2.0 §4)**: retain the copyright
  notice, provide a copy of the Apache 2.0 license, and **state that
  changes were made** to any redistributed modified files — this is a
  real, distinct requirement of this license (not present in MIT), see
  "What we changed" below.
- **What we changed**: pulled via `src/data/sources/prodnull.py` and
  schema-remapped into this project's unified `Example` schema
  (`content_source_type`, `candidate_content`, `label`, `technique`,
  `source`, `notes`). The upstream card's 24-category injection taxonomy
  is not present as a column in this release, so every malicious row was
  assigned `InjectionTechnique.OTHER` rather than a derived category — a
  labeling decision made by this project, not present in the source. No
  sampling — all rows pulled were kept.
- **Final row count in `data/processed/curated_pool.jsonl`**: **5,671**
  (2,916 malicious / 2,755 benign) — matches the pull count exactly;
  nothing was dropped between pull and composition for this source.

---

## `leolee99/NotInject`

- **Link**: https://huggingface.co/datasets/leolee99/NotInject
- **License**: MIT (confirmed directly from the dataset card).
- **Attribution notice required (MIT)**: retain the copyright notice and
  license text in copies or substantial portions of the software/data. No
  notice-of-changes obligation (unlike Apache 2.0) and no share-alike
  obligation (unlike CC BY-SA 4.0) — MIT's requirement is the lightest of
  the three licenses in this file.
- **What we changed**: pulled via `src/data/sources/notinject.py` from all
  three HF splits (`NotInject_one`/`_two`/`_three`, 113 rows each, keyed by
  trigger-word count) and schema-remapped into this project's `Example`
  schema — every row mapped to `Label.BENIGN`, `technique=None` (source
  data is entirely benign by construction, so no relabeling of malicious
  content was involved). No sampling — all rows pulled were kept.
- **Final row count in `data/processed/curated_pool.jsonl`**: **339**
  (0 malicious / 339 benign) — matches the pull count exactly.

---

## `MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT`

- **Link**: https://huggingface.co/datasets/MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT
- **License**: CC BY-SA 4.0, as claimed by the dataset card (blanket
  claim). **Important caveat, found during development and carried
  forward here rather than re-derived**: ~48.3% of this dataset's
  *malicious* half was found to be
  content-pattern-matched to BIPIA's WebQA/Summarization domains (XSum,
  NewsQA), which Microsoft's own `microsoft/BIPIA` README excludes from
  redistribution "due to license issues" — the blanket CC BY-SA 4.0 claim
  likely does not correctly cover that content. **This is moot for what's
  actually in this repo**, because of the next point.
- **Only the benign half was used — the malicious half is excluded
  entirely, not merely capped.** This composition does not use
  MAlmasabi's malicious rows at all. The license-provenance concern above
  is specifically about the malicious half; MAlmasabi's benign half is
  independently GPT-4o-mini-generated content (only ~3.2% overlaps
  BIPIA's actual clean corpus), so that specific concern does not carry
  over to the rows actually committed here.
- **Attribution notice required (CC BY-SA 4.0)**: attribution to the
  dataset (name, link, indication of changes made) **and** share-alike —
  any adaptation of this content must be distributed under CC BY-SA 4.0
  (or a license the CC organization has designated as compatible). This
  is the strictest obligation among the four sources in this file — see
  "Share-alike consequence" section below.
- **What we changed**: pulled via `src/data/sources/malmasabi.py`
  (70,000 rows total pulled). Of the 35,000-row benign half: a coherence
  filter (dropping rows whose `user_intent` references a capitalized
  entity absent from `context`) was applied first, passing 22,622 of
  35,000; a domain-stratified sample of 2,521 was then drawn from that
  coherent remainder, preserving its natural domain mix (unclassified/
  table/code/email) rather than sampling uniformly. Rows were
  schema-remapped into this project's `Example` schema. The malicious
  half (35,000 rows) was pulled but entirely excluded from composition.
- **Final row count in `data/processed/curated_pool.jsonl`**: **2,521**,
  all benign, 0 malicious. **Discrepancy from the pulled pool size,
  flagged rather than reconciled**: the pulled pool for this source was
  35,000 malicious / 35,000 benign — the 2,521 figure here reflects the
  coherence-filtered, domain-stratified, malicious-excluded *composition*
  result, a downstream narrowing from the pull, not a contradiction of
  the pull-count itself.

---

## `microsoft/BIPIA`

- **Link**: https://github.com/microsoft/BIPIA (arXiv:2312.14197)
- **License**: **Mixed, not a single license** — confirmed by reading the
  actual repo's LICENSE file directly (`docs/CITATIONS.md` C7), not
  assumed from a secondary source:
  - Code (the `bipia` Python package, including the insertion primitives
    reused by `src/data/sources/bipia.py`): **MIT**, copyright Microsoft
    Corporation.
  - **EmailQA** content (sourced from OpenAI Evals invoices): **MIT**.
  - **TableQA** content (sourced from WikiTableQuestions): **CC BY-SA
    4.0**.
  - **CodeQA** content (sourced from Stack Exchange): **CC BY-SA 4.0**.
- **What we changed**: `src/data/sources/bipia.py` reuses *only*
  `bipia.data.utils`'s `insert_end`/`insert_start`/`insert_middle`
  insertion primitives from the original repo — not the `AutoPIABuilder`/
  `QAPIABuilder` classes or `construct_samples()` — and implements its own
  pairing/combination loop and schema mapping (`docs/CITATIONS.md` C7
  explains why: the builder classes target constructing benchmark chat
  prompts, not matched clean/poisoned rows for a classifier dataset).
  Malicious side: **not the full 41,250-row generative capacity** — a
  stratified sample by `(domain, template, insertion_position)` at cap=6
  per stratum (600 strata × 6 = 3,600 rows). Benign side: **not sampled**
  — all 200 original test-split contexts retained unconditionally (the
  malicious sample's anchor set) plus 701 deduplicated train-split
  additions = 901 rows, the full offer from BIPIA's corpus under a
  corrected dedup rule applied during composition. Attack-template
  category was mapped to this project's `InjectionTechnique` taxonomy
  (decision D22).
- **Final row count in `data/processed/curated_pool.jsonl`**: **4,501**
  total (3,600 malicious / 901 benign), split by domain (and therefore by
  which of the two licenses above actually governs each row):

  | Domain | License | Malicious | Benign | Total |
  |---|---|---|---|---|
  | EmailQA | MIT | 1,350 | 84 | 1,434 |
  | TableQA | CC BY-SA 4.0 | 1,350 | 717 | 2,067 |
  | CodeQA | CC BY-SA 4.0 | 900 | 100 | 1,000 |
  | **Total** | | **3,600** | **901** | **4,501** |

  So of BIPIA's 4,501 committed rows: **1,434 are MIT-governed**, **3,067
  are CC BY-SA 4.0-governed**.
- **Attribution notice required**: MIT terms (copyright/license retention)
  for the EmailQA-sourced 1,434 rows; CC BY-SA 4.0 terms (attribution +
  share-alike) for the TableQA/CodeQA-sourced 3,067 rows.

---

## Share-alike consequence for `data/processed/` (CC BY-SA 4.0)

**`data/processed/` is mixed-license by construction, not uniformly
covered by whatever `LICENSE` governs this repo's code and self-authored
examples.** CC BY-SA 4.0's share-alike clause is stricter than Apache
2.0's notice-of-changes clause, which is in turn stricter than MIT's
plain attribution requirement — these are not interchangeable, and rows
governed by CC BY-SA 4.0 cannot be folded under a different blanket
license for redistribution:

- **2,521 rows** (MAlmasabi, all benign) are CC BY-SA 4.0.
- **3,067 rows** (BIPIA TableQA/CodeQA-sourced) are CC BY-SA 4.0.
- **5,588 of 13,032 committed rows (42.9%) are CC BY-SA 4.0-governed and
  must remain so** in any redistribution of `data/processed/` — this
  applies to the dataset folder specifically, independent of how the
  code in `src/`/`scripts/` is licensed.

The remaining 7,444 rows (prodnull 5,671 Apache 2.0 + NotInject 339 MIT +
BIPIA EmailQA 1,434 MIT) carry lighter, non-share-alike obligations
(attribution and, for prodnull specifically, notice-of-changes).

---

## Self-authored / adapted examples (not a third-party license entry)

`data/processed/self_authored.jsonl` — **3 rows** (1 malicious, 2 benign)
as of this writing — is original content produced for this project
(Track 0B) and is covered by this repo's own primary `LICENSE`, not a
third-party notice. Included here only for a complete row-count
accounting in one place, not as a license claim.

---

## Total reconciliation

| Source | Rows | License |
|---|---|---|
| prodnull | 5,671 | Apache 2.0 |
| NotInject | 339 | MIT |
| MAlmasabi (benign only) | 2,521 | CC BY-SA 4.0 |
| BIPIA (EmailQA) | 1,434 | MIT |
| BIPIA (TableQA + CodeQA) | 3,067 | CC BY-SA 4.0 |
| **Third-party subtotal** | **13,032** | mixed |
| Self-authored (this repo's `LICENSE`) | 3 | — |
| **Total, `data/processed/`** | **13,035** | — |

**Cross-checked against `data/processed/train.jsonl` + `val.jsonl` +
`test.jsonl`** (10,430 + 1,293 + 1,309 = 13,032, the split of
`curated_pool.jsonl`) **+ `self_authored.jsonl`'s 3 rows = 13,035.**
Reconciles exactly — no discrepancy found against the split files.

**Not independently re-verified while writing this file** (flagged rather
than guessed): whether `data/processed/train.jsonl`'s 10,430 figure and
other internal references to "10,430 curated" rows are describing the
same split consistently everywhere that number is cited — they appear
consistent everywhere checked, but this file does not constitute a fresh
independent audit of every citation of that number elsewhere in the repo.
