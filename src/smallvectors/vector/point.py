from smallvectors.core import Immutable, Mutable
from smallvectors.vector.linear import LinearAny
from smallvectors.vector.vec import mVec, Vec


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


# Vector conversions
def aspoint(obj):
    """
    Return object as an immutable point.
    """

    if isinstance(obj, Point):
        return obj
    else:
        return Point(*obj)


def asmpoint(obj):
    """
    Return object as a mutable point.
    """

    if isinstance(obj, mPoint):
        return obj
    else:
        return mPoint(*obj)


def asapoint(obj):
    """
    Return object as a mutable or immutable point.

    Non-Point objects are converted to immutable points.
    """

    if isinstance(obj, PointAny):
        return obj
    else:
        return Point(*obj)
