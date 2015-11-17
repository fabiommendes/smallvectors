import pytest
from smallvectors import Affine


def test_affine_shape():
    T = Affine[2, float]
    assert T.size == 6


if __name__ == '__main__':
    pytest.main('test_affine.py -q')



