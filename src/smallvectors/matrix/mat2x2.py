from generic import convert

from smallvectors import FlatView, Vec
from smallvectors.matrix.square import SquareMixin


class Mat2x2Mixin(SquareMixin):
    __slots__ = ('_a', '_b', '_c', '_d')

    @classmethod
    def from_flat(cls, data, copy=True, dtype=None):
        if cls.__concrete__ and dtype is None:
            a, b, c, d = data
            dtype = cls.dtype
            new = object.__new__(cls)
            new._a = convert(a, dtype)
            new._b = convert(b, dtype)
            new._c = convert(c, dtype)
            new._d = convert(d, dtype)
            return new
        return super(Mat2x2Mixin, cls).from_flat(data, copy=copy, dtype=dtype)

    @property
    def flat(self):
        return FlatView(self)

    def __init__(self, row1, row2):
        dtype = self.dtype
        a, b = row1
        c, d = row2
        self._a = convert(a, dtype)
        self._b = convert(b, dtype)
        self._c = convert(c, dtype)
        self._d = convert(d, dtype)

    def __flatiter__(self):
        yield self._a
        yield self._b
        yield self._c
        yield self._d

    def __flatgetitem__(self, i):
        if i == 0:
            return self._a
        elif i == 1:
            return self._b
        elif i == 2:
            return self._c
        elif i == 3:
            return self._d
        else:
            raise IndexError(i)

    def __flatsetitem__(self, i, value):
        if i == 0:
            self._a = convert(value, self.dtype)
        elif i == 1:
            self._b = convert(value, self.dtype)
        elif i == 2:
            self._c = convert(value, self.dtype)
        elif i == 3:
            self._d = convert(value, self.dtype)
        else:
            raise IndexError(i)

    def __iter__(self):
        vec = Vec[2, self.dtype]
        yield vec(self._a, self._b)
        yield vec(self._c, self._d)

    def cols(self):
        vec = Vec[2, self.dtype]
        yield vec(self._a, self._c)
        yield vec(self._b, self._d)

    def items(self):
        yield (0, 0), self._a
        yield (0, 1), self._b
        yield (1, 0), self._c
        yield (1, 1), self._d

    def det(self):
        return self._a * self._d - self._b * self._c

    def trace(self):
        return self._a + self._d

    def diag(self):
        return Vec[2, self.dtype](self._a, self._d)

    def inv(self):
        det = self.det()
        return self._matrix([self._d / det, -self._b / det],
                            [-self._c / det, self._a / det])

    def transpose(self):
        return self.__class__([self._a, self._c],
                              [self._b, self._d])

    def eigenpairs(self):
        a, b, c, d = self.flat
        l1 = (d + a + self._sqrt(d * d - 2 * a * d + a * a + 4 * c * b)) / 2
        l2 = (d + a - self._sqrt(d * d - 2 * a * d + a * a + 4 * c * b)) / 2

        try:
            v1 = Vec(b / (l1 - a), 1)
        except ZeroDivisionError:
            v1 = Vec(1, 0)
        try:
            v2 = Vec(b / (l2 - a), 1)
        except ZeroDivisionError:
            v2 = Vec(1, 0)

        return [(l1, v1.normalize()), (l2, v2.normalize())]
