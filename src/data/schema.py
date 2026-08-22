"""
Unified example schema for the IPI detector.

Design rationale: docs/DECISIONS.md, D8 and D8a.

- `content_source_type` lets the same schema later cover the malicious
  tool-call surface (D9, stretch) without a schema migration.
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
    BENIGN = "benign"
    MALICIOUS = "malicious"


class InjectionTechnique(str, Enum):
    """
    Used only when label == MALICIOUS. Populated for both the original
    training set and red-team examples, so eval can be segmented by
    technique category rather than reported as one aggregate number
    (docs/DECISIONS.md, D10).
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
        assert ex.technique is not None, "malicious examples must be labeled with a technique (D10 segmentation)"
