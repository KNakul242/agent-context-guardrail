import pytest

from src.data.schema import ContentSourceType, Example, InjectionTechnique, Label, validate


def _example(**overrides) -> Example:
    defaults = dict(
        example_id="ex1",
        content_source_type=ContentSourceType.TOOL_OUTPUT,
        candidate_content=["some tool output"],
        label=Label.BENIGN,
    )
    defaults.update(overrides)
    return Example(**defaults)


def test_empty_candidate_content_window_rejected():
    ex = _example(candidate_content=[])
    with pytest.raises(AssertionError):
        validate(ex)


def test_benign_example_with_technique_rejected():
    ex = _example(label=Label.BENIGN, technique=InjectionTechnique.DIRECT_OVERRIDE)
    with pytest.raises(AssertionError):
        validate(ex)


def test_benign_example_without_technique_accepted():
    ex = _example(label=Label.BENIGN, technique=None)
    validate(ex)  # should not raise


def test_malicious_tool_output_without_technique_rejected():
    ex = _example(
        content_source_type=ContentSourceType.TOOL_OUTPUT,
        label=Label.MALICIOUS,
        technique=None,
    )
    with pytest.raises(AssertionError):
        validate(ex)


def test_malicious_tool_output_with_technique_accepted():
    ex = _example(
        content_source_type=ContentSourceType.TOOL_OUTPUT,
        label=Label.MALICIOUS,
        technique=InjectionTechnique.FAKE_SYSTEM_TAG,
    )
    validate(ex)  # should not raise


def test_malicious_tool_call_with_technique_rejected():
    """D21: InjectionTechnique categorizes injection *method*, meaningless
    for TOOL_CALL examples where MALICIOUS means action-danger."""
    ex = _example(
        content_source_type=ContentSourceType.TOOL_CALL,
        label=Label.MALICIOUS,
        technique=InjectionTechnique.DIRECT_OVERRIDE,
    )
    with pytest.raises(AssertionError):
        validate(ex)


def test_malicious_tool_call_without_technique_accepted():
    ex = _example(
        content_source_type=ContentSourceType.TOOL_CALL,
        label=Label.MALICIOUS,
        technique=None,
    )
    validate(ex)  # should not raise
