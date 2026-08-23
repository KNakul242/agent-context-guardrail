import pytest
import torch

from src.model.aggregation import pool


def test_window_of_one_is_identity():
    """The whole point of D8a's default window size of 1 is that it's
    exactly equivalent to a stateless single-input baseline -- pooling a
    single embedding must return that embedding unchanged, not an
    approximation of it."""
    embedding = torch.tensor([1.0, 2.0, 3.0])
    result = pool([embedding], strategy="mean")
    assert torch.equal(result, embedding)


def test_mean_pooling_averages_elementwise():
    embeddings = [
        torch.tensor([1.0, 0.0, 3.0]),
        torch.tensor([3.0, 2.0, 1.0]),
    ]
    result = pool(embeddings, strategy="mean")
    assert torch.allclose(result, torch.tensor([2.0, 1.0, 2.0]))


def test_max_pooling_takes_elementwise_max():
    embeddings = [
        torch.tensor([1.0, 5.0, 3.0]),
        torch.tensor([4.0, 2.0, 3.5]),
    ]
    result = pool(embeddings, strategy="max")
    assert torch.allclose(result, torch.tensor([4.0, 5.0, 3.5]))


def test_empty_window_rejected():
    """Mirrors schema.validate()'s 'candidate_content window cannot be
    empty' invariant -- this module should not silently produce a
    zero/NaN vector for an empty window."""
    with pytest.raises(AssertionError):
        pool([], strategy="mean")


def test_unknown_strategy_rejected():
    embedding = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        pool([embedding], strategy="attention")
