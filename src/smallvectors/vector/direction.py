from smallvectors.vector.vec_2d import Vec2D
from smallvectors.vector.vec import Vec

__all__ = ['Direction', 'asdirection']


class Direction(Vec):
    """
    Direction is an immutable Vec with unitary euclidean length and represents a
    direction in euclidian space
    """

    __slots__ = ()

    def is_null(self):
        """Always False for Direction objects"""

        return False

    def is_unity(self, norm=None, tol=1e-6):
        """Always True for Direction objects"""

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
        """Return a normalized version of vector"""

        return self


class Direction2D:
    """
    A 2-dimensional direction/unity vector
    """

    __slots__ = ()

    def __init__(self, x, y):
        norm = self._sqrt(x * x + y * y)
        if norm == 0:
            raise ValueError('null vector does not define a valid direction')

        self._x = x / norm
        self._y = y / norm

    def irotate(self, theta):
        """
        Rotate vector by an angle theta around origin.
        """

        x, y = self
        cos_t, sin_t = self._cos(theta), self._sin(theta)
        new = Vec2D.__new__(Direction2D, x, y)
        new.x = x * cos_t - y * sin_t
        new.y = x * sin_t + y * cos_t
        return new


#
# Direction conversions
#
def asdirection(v):
    """Return the argument as a Direction instance."""

    if isinstance(v, Direction):
        return v
    else:
        return Direction(*v)
