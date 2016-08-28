from generic import convert

from smallvectors import FlatView, Normed
from smallvectors.vector.linear import LinearAny
from smallvectors.vector.vec import _assure_mutable_set_coord


class VecND(LinearAny, Normed):
    """
    Base class with overrides for any dimension
    """

    @property
    def flat(self):
        return FlatView(self)

    def __flatiter__(self):
        return iter(self)

    def __flatlen__(self):
        return len(self)

    def __flatgetitem__(self, idx):
        return self[idx]

    def __flatsetitem__(self, idx, value):
        self[idx] = value


class Vec0D(VecND):
    """
    0D Vectors (probably even less necessary than Vec1D :P)
    """

    __slots__ = ()

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())

    def __getitem__(self, idx):
        raise IndexError(idx)


class Vec1D(VecND):
    """
    1D Vectors (is this necessary?)
    """
    __slots__ = ('_x',)

    # noinspection PyMissingConstructor
    def __init__(self, x):
        self._x = convert(x, self.dtype)

    def __len__(self):
        return 1

    def __iter__(self):
        yield self._x

    def __getitem_simple__(self, idx):
        assert idx in (0, -1)
        return self._x

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        _assure_mutable_set_coord(self)
        self._x = convert(value, self.dtype)

    x0 = x
