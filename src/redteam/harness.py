"""
Red-team harness skeleton (Track 0A scaffolding; full seed corpus and
escalation loop are Phase 1 scope, docs/specs/IMPLEMENTATION_PLAN.md).

Structure only, deliberately: this module defines the interface a trained
detector plugs into (predict_fn: str -> Label) and the deterministic parts
around it (recording results, segmenting bypass rate by category, harvesting
bypasses into retraining data) - all fully testable now, without a trained
model or any external API, by injecting a stub predict_fn in tests. What
this module does NOT implement yet is the actual escalation loop (a frozen
strong free-tier model proposing mutations of successful attacks,
Maatphor-style, per the plan) - that needs a real trained model and a live
API call, so it isn't the kind of thing you can meaningfully TDD before
Phase 1 exists; see escalate() below for the stub.
"""

from dataclasses import dataclass
from typing import Callable, List

from src.data.schema import ContentSourceType, Example, InjectionTechnique, Label
from src.eval.metrics import bypass_rate, segment_by_category

PredictFn = Callable[[str], Label]


@dataclass
class RedTeamSeed:
    seed_id: str
    technique: InjectionTechnique
    content: str


@dataclass
class RedTeamResult:
    seed: RedTeamSeed
    bypassed: bool  # True == detector said BENIGN on genuinely malicious content


def run_seeds(seeds: List[RedTeamSeed], predict_fn: PredictFn) -> List[RedTeamResult]:
    """Run every seed through predict_fn once and record whether it bypassed
    (predict_fn wrongly said BENIGN) or was caught (correctly said
    MALICIOUS) -- every seed here is malicious by construction, so there's
    no true-negative case to track."""
    results = []
    for seed in seeds:
        predicted = predict_fn(seed.content)
        results.append(RedTeamResult(seed=seed, bypassed=(predicted == Label.BENIGN)))
    return results


def bypass_rate_by_technique(results: List[RedTeamResult]) -> dict:
    """D10: bypass rate reported per InjectionTechnique category, never one
    aggregate ASR number. Reuses eval.metrics.segment_by_category rather
    than reimplementing the same grouping logic a second time here."""
    y_true = [1 for _ in results]  # every seed is malicious by construction
    y_pred = [0 if r.bypassed else 1 for r in results]  # 0 == predicted benign, i.e. missed
    categories = [r.seed.technique.value for r in results]
    return segment_by_category(y_true, y_pred, categories, metric_fn=bypass_rate)


def harvest_bypasses(results: List[RedTeamResult]) -> List[Example]:
    """Confirmed bypasses become malicious Examples flagged is_redteam=True,
    ready to append to data/redteam/ and retrain on (Phase 1's harvest-retrain
    step). Only bypassed seeds are harvested -- a seed the detector already
    catches teaches the retrain nothing new."""
    return [
        Example(
            example_id=f"redteam-{r.seed.seed_id}",
            content_source_type=ContentSourceType.TOOL_OUTPUT,
            candidate_content=[r.seed.content],
            label=Label.MALICIOUS,
            technique=r.seed.technique,
            source="redteam",
            is_redteam=True,
            notes=f"harvested bypass; seed_id={r.seed.seed_id}",
        )
        for r in results
        if r.bypassed
    ]


def escalate(seed: RedTeamSeed, prior_results: List[RedTeamResult]) -> str:
    """Stub for the LLM-red-teamer escalation loop (Phase 1): given a seed
    and how prior rounds' mutations of it fared, call a frozen strong
    free-tier model to propose the next mutation, Maatphor-style. Needs a
    trained detector to test mutations against and a live API call to
    generate them - neither exists yet at Phase 0, so this is intentionally
    unimplemented rather than faked with a placeholder that would pass
    trivially. See docs/specs/IMPLEMENTATION_PLAN.md Phase 1."""
    raise NotImplementedError("escalation loop is Phase 1 scope, not Track 0A scaffolding")
