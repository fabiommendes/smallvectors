from .vec import Vec


class Vec4(Vec):
    """
    A 4D vector.
    """

    __slots__ = ('_x', '_y', '_z', '_w')

    size = 4
    shape = (4,)

    x0 = x = property(lambda self: self._x)
    x1 = y = property(lambda self: self._y)
    x2 = z = property(lambda self: self._z)
    x3 = w = property(lambda self: self._w)

    @classmethod
    def from_flat(cls, data):
        x, y, z, w = data
        return Vec4(x, y, z, w)

    def __init__(self, x, y, z, w):
        self._x = x + 0.0
        self._y = y + 0.0
        self._z = z + 0.0
        self._w = w + 0.0

    def __len__(self):
        return 4

    def __iter__(self):
        yield self._x
        yield self._y
        yield self._z
        yield self._w

    def __getitem__(self, idx):
        if idx in (0, -4):
            return self._x
        elif idx in (1, -3):
            return self._y
        elif idx in (2, -2):
            return self._z
        elif idx in (3, -1):
            return self._w
        elif isinstance(idx, slice):
            return [self._x, self._y, self._z, self._w][idx]
        else:
            raise IndexError(idx)

    def __eq__(self, other):
        if isinstance(other, (Vec4, list, tuple)):
            try:
                x, y, z, w = other
            except ValueError:
                return False
            return (self._x == x and self._y == y and self._z == z and
                    self._w == w)
        return NotImplemented

    def copy(self, x=None, y=None, z=None, w=None, **kwargs):
        if kwargs:
            return super().copy(x=x, y=y, z=z, **kwargs)
        if x is None:
            x = self._x
        if y is None:
            y = self._y
        if z is None:
            z = self._z
        if w is None:
            w = self._w
        return Vec4(x, y, z, w)
