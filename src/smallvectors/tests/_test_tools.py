import pytest
from smallvectors.utils import flatten


def test_flatten():
    assert flatten([1, 2]) == ([1, 2], 2)
    assert flatten([[1, 2]]) == ([1, 2], 1, 2)
    assert flatten([[1], [2]]) == ([1, 2], 2, 1)
    assert flatten([[1, 2], [3, 4], [5, 6]]) == ([1, 2, 3, 4, 5, 6], 3, 2)
