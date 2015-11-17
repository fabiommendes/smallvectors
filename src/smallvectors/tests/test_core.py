import pytest
from smallvectors.tools import shape, commonbase


def test_shape_function():
    assert shape([[1, 2], [3, 4], [5, 6]]) == (3, 2)


def test_commonbase_function():
    assert commonbase(int, int) == int
    assert commonbase(int, int, int) == int


if __name__ == '__main__':
    pytest.main('test_core.py -q')