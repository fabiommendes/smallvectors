import pytest
from smallvectors.core import Flat, mFlat, FlatView


@pytest.fixture
def HasView():
    class HasView:
        size = 4

        def __init__(self):
            self.data = [1, 2, 3, 4]
            self.flat = FlatView(self)

        def __flatiter__(self):
            return iter(self.data)

        def __flatgetitem__(self, i):
            return self.data[i]

    return HasView


@pytest.fixture
def hasview():
    return HasView()()


#
# Unit tests
#
def test_view_getitem(hasview):
    assert hasview.flat[0] == 1


def test_view_getslice(hasview):
    assert hasview.flat[0:2] == [1, 2]


def test_view_getslice_step(hasview):
    assert hasview.flat[0::2] == [1, 3]


if __name__ == '__main__':
    pytest.main('test_flatobject.py -q')