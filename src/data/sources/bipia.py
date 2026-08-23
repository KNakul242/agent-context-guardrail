"""
microsoft/BIPIA -> Example schema mapping (docs/DATA_SOURCES.md, D19;
docs/CITATIONS.md C7 for what's borrowed vs. our own here).

Source: third_party/BIPIA (git clone, MIT for the code; per-task content
license documented in DATA_SOURCES.md). Highest-quality source in the pool
because it gives genuine matched pairs: the same context, with and without
an injected attack string, rather than two separately-sourced pools.

Deliberate scope decision on how much of BIPIA's own library we reuse:
`bipia.data`'s builder classes (AutoPIABuilder, QAPIABuilder, CodeIPIABuilder,
...) are designed to build *chat prompts* for benchmarking an LLM under test
(system_prompt_template / user_prompt_template / construct_response) - that's
a different job than ours, which is just "give me matched clean/poisoned
content pairs" for a classifier dataset. Forcing our data through their
prompt-construction machinery would mean fighting an abstraction built for a
different purpose (Occam's razor, DEVELOPMENT_RULES.md).

The one genuinely non-trivial, worth-reusing piece is the insertion logic in
bipia.data.utils - insert_end/insert_start/insert_middle - especially
insert_middle, which uses nltk's PunktSentenceTokenizer to insert the attack
string at a random *sentence* boundary rather than a random character offset,
so the poisoned text stays syntactically plausible. We import only that
utils module (via sys.path, not `pip install -e .`) so we don't drag in
BIPIA's full dependency list (vllm, deepspeed, fschat, wandb - all for
running inference against candidate LLMs, none of which we need here and
some of which don't install cleanly on Mac/MPS anyway).

Category -> InjectionTechnique mapping (D22, logged in docs/DECISIONS.md):
BIPIA's attack templates are grouped by *attacker objective* (e.g. "Task
Automation", "Scams & Fraud", "Data Eavesdropping") - what the injected
instruction asks the agent to do - not by injection *method*, which is what
our InjectionTechnique enum actually categorizes. Inspecting the actual
attack strings (e.g. "Write a script to rename all files...") shows they're
all the same method: a bare, undisguised imperative sentence inserted into
the content, hoping the agent follows it as if it were a legitimate
instruction. That's what DIRECT_OVERRIDE is meant to capture at the broad
level, even though its inline comment's example ("ignore previous
instructions and...") is a narrower sub-case - so every objective-based
category defaults to DIRECT_OVERRIDE. A separate handful of categories
*are* method-based (they obfuscate/encode the payload rather than state it
plainly) and map onto our existing ENCODING_OBFUSCATION /
LOW_RESOURCE_LANGUAGE buckets. "Language Translation" -> LOW_RESOURCE_LANGUAGE
is an approximation worth flagging in the write-up: BIPIA translates into
whatever language the template specifies (e.g. Spanish), not specifically a
*low-resource* one - the enum just doesn't have a plain "translated" bucket,
and this is the closest existing fit rather than a perfect one.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import List, Union

import jsonlines

from src.data.schema import ContentSourceType, Example, InjectionTechnique, Label

# Point at the sibling third_party/BIPIA clone (repo_root/third_party/BIPIA).
# parents[3] from this file: sources -> data -> src -> repo_root.
_BIPIA_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "BIPIA"
if str(_BIPIA_ROOT) not in sys.path:
    sys.path.insert(0, str(_BIPIA_ROOT))

# Importing bipia.data.utils directly (not the bipia.data.__init__ builder
# classes) is what lets us skip installing BIPIA's full dependency list -
# see the module docstring above.
from bipia.data.utils import insert_end, insert_middle, insert_start  # noqa: E402

INSERT_FNS = {"end": insert_end, "start": insert_start, "middle": insert_middle}

RAW_DIR = Path("data/raw/bipia")
TASKS = ["email", "table", "code"]
# Fixed seed so the materialized raw output is reproducible from this exact
# call, per DEVELOPMENT_RULES.md's Definition of Done ("seed fixed, exact
# run command documented").
PULL_SEED = 42


def pull_raw(out_dir: Path = RAW_DIR, seed: int = PULL_SEED) -> None:
    """Materialize the full matched-pair set (all 3 tasks x all 3 insert
    positions x every attack template) from BIPIA's own test-split files.

    This writes the *entire* generative capacity, not a training-ready
    sample - e.g. code alone produces 7,500 poisoned rows from just 50
    clean contexts x 50 attacks x 3 positions. Deliberately not curated
    down here: which subset/ratio actually goes into data/processed is the
    composition decision D13 has parked (docs/DECISIONS.md), not something
    to pre-empt at the raw-pull stage. This step only needs to prove the
    wrapper produces genuine matched pairs (Track 0A's Definition of Done),
    and stage everything so that decision can be made from real numbers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        attacks_file = f"third_party/BIPIA/benchmark/{'code' if task == 'code' else 'text'}_attack_test.json"
        result = materialize_pairs(
            task=task,
            contexts=f"third_party/BIPIA/benchmark/{task}/test.jsonl",
            attacks=attacks_file,
            insert_fn_names=["end", "start", "middle"],
            seed=seed,
        )
        with open(out_dir / f"{task}_clean.jsonl", "w") as f:
            for row in result["clean"]:
                f.write(json.dumps(row) + "\n")
        with open(out_dir / f"{task}_poisoned.jsonl", "w") as f:
            for row in result["poisoned"]:
                f.write(json.dumps(row) + "\n")


def load_examples(raw_dir: Path = RAW_DIR):
    """Read pull_raw()'s output back and map every row to an Example."""
    for task in TASKS:
        with open(raw_dir / f"{task}_clean.jsonl") as f:
            for i, line in enumerate(f):
                yield map_pair_to_examples(json.loads(line), task=task, kind="clean", row_index=i)
        with open(raw_dir / f"{task}_poisoned.jsonl") as f:
            for i, line in enumerate(f):
                yield map_pair_to_examples(json.loads(line), task=task, kind="poisoned", row_index=i)


def _normalized_hash(context) -> str:
    text = "\n".join(context) if isinstance(context, list) else context
    return hashlib.md5(" ".join(text.split()).lower().encode()).hexdigest()


def benign_corpus_for_task(task: str, test_contexts, train_contexts, seed: int = 0) -> List[dict]:
    """The corrected benign pull (docs/DECISIONS.md D25): every test-split
    document is retained unconditionally -- they're the document-identity
    anchors the malicious stratified sample (pull_raw()'s poisoned output)
    is built from, referenced by clean_context_id, not by content. Only
    train-split additions are deduplicated (normalized-hash), against the
    full retained-test set plus whatever's already been added from train.

    The bug this replaces deduplicated test-split rows against each other
    too, which silently dropped some of the 200 anchor documents whenever
    two test rows happened to share identical content -- orphaning every
    malicious row derived from a dropped ID (D25 found 354/3,600, 9.8%).

    test_contexts/train_contexts: same accepted shapes as
    materialize_pairs()'s `contexts` param (a path or an already-loaded
    list), so this is testable with tiny in-memory fixtures and callable
    against the real BIPIA files identically.
    """
    test_result = materialize_pairs(task=task, contexts=test_contexts, attacks={}, insert_fn_names=["end"], seed=seed)
    train_result = materialize_pairs(task=task, contexts=train_contexts, attacks={}, insert_fn_names=["end"], seed=seed) if train_contexts else {"clean": []}

    retained = list(test_result["clean"])
    seen_hashes = {_normalized_hash(row["context"]) for row in retained}

    next_id = max((row["clean_context_id"] for row in retained), default=-1) + 1
    for row in train_result["clean"]:
        key = _normalized_hash(row["context"])
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        row = dict(row)
        row["clean_context_id"] = next_id
        next_id += 1
        retained.append(row)

    return retained


def full_benign_corpus(seed: int = 0) -> dict:
    """benign_corpus_for_task() run against the real BIPIA repo files for
    all 3 tasks -- what scripts/build_curated_pool.py and the real
    train/val/test split materialization both call."""
    return {
        task: benign_corpus_for_task(
            task,
            test_contexts=f"third_party/BIPIA/benchmark/{task}/test.jsonl",
            train_contexts=f"third_party/BIPIA/benchmark/{task}/train.jsonl",
            seed=seed,
        )
        for task in TASKS
    }

# Categories whose attack templates obfuscate/transform the payload itself,
# rather than stating a plain instruction - these get a specific technique
# instead of the DIRECT_OVERRIDE default. See the module docstring (D22).
_ENCODING_CATEGORIES = {"Base Encoding", "Substitution Ciphers", "Reverse Text", "Emoji Substitution"}


def category_to_technique(category: str) -> InjectionTechnique:
    """Map one of BIPIA's attacker-objective category labels onto our
    injection-METHOD taxonomy. See D22 in docs/DECISIONS.md for the full
    reasoning - short version: most BIPIA categories describe what the
    attack wants, not how it's phrased, and how-it's-phrased is uniformly
    "plain embedded imperative" (DIRECT_OVERRIDE) except for the explicit
    encoding/translation categories below."""
    if category in _ENCODING_CATEGORIES:
        return InjectionTechnique.ENCODING_OBFUSCATION
    if category == "Language Translation":
        return InjectionTechnique.LOW_RESOURCE_LANGUAGE
    return InjectionTechnique.DIRECT_OVERRIDE


def _extract_context_text(task: str, row: dict) -> str:
    # Only the "code" task stores context/code/error as list[str] (one
    # entry per line) rather than an already-joined string - mirrors what
    # CodeIPIABuilder.construct_samples does before insertion.
    if task == "code":
        return "\n".join(row["context"])
    return row["context"]


def _joined_if_list(value):
    if isinstance(value, list):
        return "\n".join(value)
    return value


def materialize_pairs(
    task: str,
    contexts: Union[str, Path, list],
    attacks: Union[str, Path, dict],
    insert_fn_names=("end", "start", "middle"),
    seed: int = 0,
) -> dict:
    """Build matched clean/poisoned pairs for one BIPIA task.

    Deliberately re-implements the combination loop instead of relying on
    BasePIABuilder.construct_samples()'s returned row order to recover which
    poisoned row came from which clean context - depending on an undocumented
    third-party iteration order to reconstruct that link would be fragile.
    Doing the loop ourselves means every poisoned row carries an explicit
    `clean_context_id` we can pair back to `result["clean"]`.

    contexts/attacks accept either a path (loaded the way BIPIA's own
    benchmark files are laid out) or already-loaded Python objects (used by
    the unit tests below, so tests don't need real BIPIA files or network).
    """
    if isinstance(contexts, (str, Path)):
        with jsonlines.open(contexts) as reader:
            contexts = list(reader.iter())
    if isinstance(attacks, (str, Path)):
        with open(attacks) as f:
            attacks = json.load(f)

    clean_rows = []
    for i, row in enumerate(contexts):
        clean_rows.append(
            {
                "clean_context_id": i,
                "context": _extract_context_text(task, row),
                "question": row.get("question"),
                "code": _joined_if_list(row.get("code")),
                "error": _joined_if_list(row.get("error")),
                "ideal": _joined_if_list(row.get("ideal")),
            }
        )

    poisoned_rows = []
    for position in insert_fn_names:
        insert_fn = INSERT_FNS[position]
        for clean in clean_rows:
            for category, prompts in attacks.items():
                for attack_index, attack_str in enumerate(prompts):
                    poisoned_context = insert_fn(clean["context"], attack_str, random_state=seed)
                    poisoned_rows.append(
                        {
                            "clean_context_id": clean["clean_context_id"],
                            "context": poisoned_context,
                            "attack_category": category,
                            "attack_index": attack_index,
                            "attack_str": attack_str,
                            "position": position,
                            "question": clean["question"],
                            "code": clean["code"],
                            "error": clean["error"],
                            "ideal": clean["ideal"],
                        }
                    )

    return {"task": task, "clean": clean_rows, "poisoned": poisoned_rows}


def _agent_task_context(task: str, row: dict) -> str:
    # email/table are QA-style (a "question" about the content); code is
    # phrased as "here's my error, fix my code" instead of a question.
    if task == "code":
        return row.get("error")
    return row.get("question")


def _sanitize_id_part(value: str) -> str:
    return str(value).replace(" ", "_").replace("&", "and")


def map_pair_to_examples(row: dict, task: str, kind: str, row_index: int) -> Example:
    """Map one row from materialize_pairs()'s "clean" or "poisoned" list to
    an Example. kind selects which: a "clean" row is always BENIGN/no
    technique, a "poisoned" row is always MALICIOUS with a technique from
    category_to_technique() - this mirrors validate()'s D21 invariant, so
    every Example produced here passes validate() unchanged."""
    agent_task_context = _agent_task_context(task, row)

    if kind == "clean":
        return Example(
            example_id=f"bipia-{task}-clean-{row['clean_context_id']}",
            content_source_type=ContentSourceType.TOOL_OUTPUT,
            candidate_content=[row["context"]],
            label=Label.BENIGN,
            agent_task_context=agent_task_context,
            technique=None,
            source="bipia",
            is_redteam=False,
            notes=f"task={task}; clean_context_id={row['clean_context_id']}",
        )

    if kind == "poisoned":
        technique = category_to_technique(row["attack_category"])
        category_id = _sanitize_id_part(row["attack_category"])
        example_id = (
            f"bipia-{task}-poisoned-{row['clean_context_id']}-"
            f"{category_id}-{row['attack_index']}-{row['position']}"
        )
        return Example(
            example_id=example_id,
            content_source_type=ContentSourceType.TOOL_OUTPUT,
            candidate_content=[row["context"]],
            label=Label.MALICIOUS,
            agent_task_context=agent_task_context,
            technique=technique,
            source="bipia",
            is_redteam=False,
            notes=(
                f"task={task}; attack_category={row['attack_category']}; "
                f"position={row['position']}; clean_context_id={row['clean_context_id']}"
            ),
        )

    raise ValueError(f"kind must be 'clean' or 'poisoned', got {kind!r}")
