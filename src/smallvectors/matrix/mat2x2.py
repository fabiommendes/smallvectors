from smallvectors import Vec, Vec2
from .square import MatSquare


class Mat2x2(MatSquare):
    __slots__ = ('_a', '_b', '_c', '_d')
    size = 2
    shape = (2, 2)
    nrows = ncols = 2

    @classmethod
    def from_flat(cls, data):
        a, b, c, d = data
        new = object.__new__(cls)
        new._a = a + 0.0
        new._b = b + 0.0
        new._c = c + 0.0
        new._d = d + 0.0
        return new

    @property
    def flat(self):
        return [self._a, self._b, self._c, self._d]

    def __init__(self, row1, row2):
        a, b = row1
        c, d = row2
        self._a = a + 0.0
        self._b = b + 0.0
        self._c = c + 0.0
        self._d = d + 0.0

    def __iter__(self):
        yield Vec2(self._a, self._b)
        yield Vec2(self._c, self._d)

    def __eq__(self, other):
        if isinstance(other, (Mat2x2, list, tuple)):
            a, b = self
            u, v = other
            return a == u and b == v
        return NotImplemented

    def cols(self):
        yield Vec2(self._a, self._c)
        yield Vec2(self._b, self._d)

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
        return Vec2(self._a, self._d)

    def inv(self):
        det = self.det()
        return type(self)([self._d / det, -self._b / det],
                          [-self._c / det, self._a / det])

    def transposed(self):
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
