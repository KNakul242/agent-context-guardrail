"""
Live LLM client for D10's automated escalation loop (Phase 1).

Mirrors scripts/example_lab_gemini.py's Gemini-via-OpenAI-compatible-endpoint
pattern for consistency with Track 0B's existing tooling -- same
secrets.env/GEMINI_API_KEY convention, same model family, same reasoning for
using Gemini over OpenRouter (the earlier "frozen strong free-tier model"
plan in IMPLEMENTATION_PLAN.md/D10, unchanged in substance -- Gemini is that
model here).

Deliberately not unit-tested: this is a live network call, same category as
every pull_raw() in src/data/sources/*.py. src/redteam/harness.py's
escalate()/run_escalation_loop() take llm_call as an injected dependency
specifically so everything else in the escalation loop IS testable with a
stub -- this module is the one real implementation of that dependency.
"""

import os
import time
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECRETS_FILE = PROJECT_ROOT / "secrets.env"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# ISSUE-10 (docs/ISSUES.md): gemini-3.7-flash's free-tier daily quota is
# only 20 requests/day -- confirmed by directly hitting
# RateLimitError/RESOURCE_EXHAUSTED (quotaId
# GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue '20')
# partway through a single ~22-seed escalation run. gemini-3.5-flash-lite's
# free tier is 1000 req/day, comfortably enough for a full run -- switched
# per direct instruction. Its per-minute limit (15/min) is tighter than the
# daily one, though, and nothing here throttled calls at all before this
# change -- see MIN_SECONDS_BETWEEN_CALLS below for why that's fixed too,
# not just the model swap.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
# 15 req/min = one call every 4s at the theoretical limit; 5s leaves margin
# without meaningfully slowing a ~20-30-call run (worth ~1-2 extra minutes
# total, not worth a fancier token-bucket scheme at this scale/time budget).
MIN_SECONDS_BETWEEN_CALLS = 5.0
# Peer-review finding: the original max_tokens=300 silently truncated
# needle-in-haystack mutations mid-generation -- that technique's whole
# point is a long carrier document (data/redteam/seeds.jsonl's own
# needle_in_haystack seeds are already several hundred tokens of filler
# alone), and IMPLEMENTATION_PLAN.md Phase 1 DoD specifically calls out
# "needle-in-haystack / long-carrier-document bypass rate" as a
# load-bearing number feeding D7's ModernBERT gate. A truncated mutation
# gets scored as a malformed attack with no error -- corrupting exactly
# that number. 4096 comfortably covers every seed category's mutations;
# still finite (an unbounded response risks one bad call consuming an
# outsized share of a query budget) but far past where truncation would
# plausibly bite.
DEFAULT_MAX_TOKENS = 4096


def build_gemini_llm_call(model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> Callable[[str], str]:
    load_dotenv(SECRETS_FILE)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(f"GEMINI_API_KEY not found in {SECRETS_FILE}")

    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    last_call_at = [None]  # mutable cell so the closure can track state across calls

    def llm_call(prompt: str) -> str:
        if last_call_at[0] is not None:
            elapsed = time.monotonic() - last_call_at[0]
            wait = MIN_SECONDS_BETWEEN_CALLS - elapsed
            if wait > 0:
                time.sleep(wait)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        last_call_at[0] = time.monotonic()
        return response.choices[0].message.content or ""

    return llm_call
