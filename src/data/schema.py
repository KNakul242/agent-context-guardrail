"""
Unified example schema for the IPI detector.

Design rationale: docs/DECISIONS.md, D8, D8a, and D21.

- `content_source_type` lets the same schema later cover the malicious
  tool-call surface (D9, stretch) without a schema migration — but see D21:
  the schema extends, the Label semantics do NOT. "Malicious" means
  something different for TOOL_OUTPUT (injection-presence) vs. TOOL_CALL
  (action-danger); read the Label enum's docstring before touching D9.
- `candidate_content` is a List[str] (a window of recent tool outputs, not a
  single string) so multi-turn aggregation (D8a) can be added later by
  changing the model's forward pass and the (isolated) aggregation function,
  without touching this schema or the data pipeline. Default window size is 1,
  which is exactly equivalent to every stateless single-input baseline we
  compare against.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ContentSourceType(str, Enum):
    TOOL_OUTPUT = "tool_output"   # primary surface (D2): file/webpage/API response the agent reads
    TOOL_CALL = "tool_call"       # stretch surface (D9): a command/action the agent is about to issue


class Label(str, Enum):
    """
    IMPORTANT (D21): this enum's meaning is NOT universal across
    `content_source_type` — it is deliberately overloaded, and that's a
    documented design decision, not an oversight:

    - content_source_type == TOOL_OUTPUT (primary surface, D2): MALICIOUS
      means "this content contains an instruction embedded to hijack the
      agent's behavior" — a content-level, injection-presence label.
    - content_source_type == TOOL_CALL (stretch surface, D9): MALICIOUS
      means "this action/command is dangerous to execute" — an
      action-safety label, orthogonal to cause (a command can be
      MALICIOUS with no injection involved anywhere in its history).

    These are two different classification tasks sharing a schema for
    engineering convenience, not one label meaning extended to a second
    surface. If D9 is ever built, train it as a separate model or
    classification head — do not pool TOOL_OUTPUT and TOOL_CALL examples
    into one binary decision boundary; that would conflate two distinct
    learning problems.
    """
    BENIGN = "benign"
    MALICIOUS = "malicious"


class InjectionTechnique(str, Enum):
    """
    Applies ONLY to content_source_type == TOOL_OUTPUT malicious examples
    (D21) — this enum categorizes injection METHOD, which has no meaning
    for TOOL_CALL (D9) examples, where "malicious" means action-danger,
    not injection-presence. A future D9 implementation will need its own
    categorization enum (e.g. by danger type: destructive, exfiltration,
    privilege-escalation) — do not reuse this one.

    Populated for both the original training set and red-team examples, so
    eval can be segmented by technique category rather than reported as
    one aggregate number (docs/DECISIONS.md, D10).
    """
    DIRECT_OVERRIDE = "direct_override"                # "ignore previous instructions and..."
    FAKE_SYSTEM_TAG = "fake_system_tag"                 # spoofed <system>/[INST]-style markers
    ROLE_REFRAME = "role_reframe"                       # "you are now..." / persona hijack
    ENCODING_OBFUSCATION = "encoding_obfuscation"       # base64 / rot13 / hex payloads
    UNICODE_OBFUSCATION = "unicode_obfuscation"         # homoglyphs, zero-width chars
    PAYLOAD_SPLIT = "payload_split"                     # split across fields/turns (window > 1 only)
    NEEDLE_IN_HAYSTACK = "needle_in_haystack"           # short payload inside long benign content
    LOW_RESOURCE_LANGUAGE = "low_resource_language"     # translated injection
    FICTIONAL_FRAMING = "fictional_framing"             # "just an example, don't worry about it"
    OTHER = "other"


@dataclass
class Example:
    example_id: str
    content_source_type: ContentSourceType
    # Window of tool outputs ending at the current one. len == 1 for the
    # baseline (non-multi-turn) configuration.
    candidate_content: list[str]
    label: Label
    # Task context the agent was given, if available/relevant (helps
    # distinguish "benign imperative language" from "instruction directed at
    # the agent" — this is the NotInject-style hard-negative axis).
    agent_task_context: Optional[str] = None
    technique: Optional[InjectionTechnique] = None
    source: str = "self_authored"          # which dataset/provenance this came from
    is_redteam: bool = False               # True if constructed during the red-team step (D10)
    notes: str = ""


def validate(ex: Example) -> None:
    assert len(ex.candidate_content) >= 1, "candidate_content window cannot be empty"
    if ex.label == Label.BENIGN:
        assert ex.technique is None, "benign examples should not carry an injection technique label"
    if ex.label == Label.MALICIOUS:
        if ex.content_source_type == ContentSourceType.TOOL_OUTPUT:
            assert ex.technique is not None, (
                "malicious TOOL_OUTPUT examples must be labeled with an InjectionTechnique "
                "(D10 segmentation) — this does not apply to TOOL_CALL examples, see D21"
            )
        elif ex.content_source_type == ContentSourceType.TOOL_CALL:
            assert ex.technique is None, (
                "InjectionTechnique does not apply to TOOL_CALL examples (D21) — "
                "'malicious' here means action-danger, not injection-presence. "
                "A D9 implementation needs its own categorization enum, not this one."
            )
