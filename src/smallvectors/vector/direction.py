from smallvectors.vector.vec_2d import Vec2D
from smallvectors.vector.vec import Vec

__all__ = ['Direction', 'asdirection']


class Direction(Vec):
    """
    Direction is an immutable Vec with unitary euclidean length and represents a
    direction in Euclidian space
    """

    __slots__ = ()

    def is_null(self):
        return False

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

    def normalize(self):
        return self


# Direction conversions
def asdirection(v):
    """
    Return the argument as a Direction instance.
    """

    if isinstance(v, Direction):
        return v
    else:
        return Direction(*v)
