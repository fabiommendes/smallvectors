from smallvectors import Immutable, Mutable, mVec, Vec
from .linear import LinearAny


class PointAny(LinearAny):
    """
    Base class for Point and mPoint.
    """

    def __add__(self, other):
        if isinstance(other, PointAny):
            raise TypeError('cannot add two point instances')
        return super().__add__(other)

    def __sub__(self, other):
        delta = super().__sub__(other)
        if isinstance(other, PointAny):
            if other.mutable:
                return mVec(delta)
            else:
                return Vec(delta)
        return delta


class Point(PointAny, Immutable):
    """
    An immutable Point type.
    """


class mPoint(PointAny, Mutable):
    """
    A mutable Point type.
    """
