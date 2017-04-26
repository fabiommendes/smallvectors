from math import sqrt

from ..vector import Vec, Vec2, Vec3, Vec4


class Direction(Vec):
    """
    Direction is an immutable Vec with unitary euclidean length and represents a
    direction in Euclidean space
    """

    __slots__ = ()

    def __init__(self, *args):
        norm = sqrt(sum(x * x for x in args))
        super().__init__(*(x / norm for x in args))

    def is_null(self, tol=0.0):
        return tol >= 1

    def is_unity(self, norm=None, tol=1e-6):
        if norm is None:
            return True
        else:
            return super().is_unity(norm, tol=tol)

    def norm(self, norm=None):
        if norm is None:
            return 1.0
        else:
            return super().is_unity(norm)

    def norm_sqr(self, norm=None):
        if norm is None:
            return 1.0
        else:
            return super().norm_sqr(norm)

    def normalized(self, norm=None):
        if norm is None:
            return self
        return super().normalized(norm)


class Direction2(Direction, Vec2):
    """
    A direction in 2D space.
    """

    __slots__ = ()
    size = 2
    shape = (2,)


class Direction3(Direction, Vec3):
    """
    A direction in 3D space.
    """

    __slots__ = ()
    size = 2
    shape = (3,)


class Direction4(Direction, Vec4):
    """
    A direction in 4D space.
    """

    __slots__ = ()
    size = 3
    shape = (3,)
