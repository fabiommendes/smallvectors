from math import sin, cos, atan2, sqrt

from .linear import Linear
from .vec import Vec, L2_NORMS


class Vec2(Vec):
    """
    A 2D immutable vector of floats.
    """

    __slots__ = ('_x', '_y')
    size = 2
    shape = (2,)
    x0 = x = property(lambda self: self._x)
    x1 = y = property(lambda self: self._y)

    @classmethod
    def from_flat(cls, data):
        x, y = data
        return Vec2(x, y)

    @classmethod
    def from_polar(cls, radius, theta=0):
        """
        Create vector from polar coordinates.
        """
        return cls(radius * cos(theta), radius * sin(theta))

    def __init__(self, x, y):
        self._x = x + 0.0
        self._y = y + 0.0

    def __len__(self):
        return 2

    def __iter__(self):
        yield self._x
        yield self._y

    def __getitem__(self, idx):
        if idx in (0, -2):
            return self._x
        elif idx in (1, -1):
            return self._y
        elif isinstance(idx, slice):
            return [self._x, self._y][idx]
        else:
            raise IndexError(idx)

    def __eq__(self, other):
        if isinstance(other, (Vec2, list, tuple, Linear)):
            try:
                x, y = other
            except ValueError:
                return False
            return self._x == x and self._y == y
        return NotImplemented

    #
    # Mathematical operations
    #
    def __add__(self, other):
        if isinstance(other, (Vec2, tuple, list)):
            x, y = other
            return Vec2(self._x + x, self._y + y)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (Vec2, tuple, list, Linear)):
            x, y = other
            return Vec2(self._x - x, self._y - y)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (Vec2, tuple, list, Linear)):
            x, y = other
            return Vec2(x - self._x, y - self._y)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Vec2(self._x * other, self._y * other)
        return NotImplemented

    __rmul__ = __mul__

    def __floordiv__(self, other):
        if isinstance(other, (float, int)):
            return Vec2(self._x // other, self._y // other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return Vec2(self._x / other, self._y / other)
        return NotImplemented

    #
    # Abstract methods
    #
    def copy(self, x=None, y=None, **kwargs):
        if kwargs:
            return super().copy(x=x, y=y, **kwargs)
        if x is None:
            x = self._x
        if y is None:
            y = self._y
        return Vec2(x, y)

    #
    # 2D specific API
    #
    def rotated_axis(self, theta, axis):
        """
        Rotate vector around given axis by the angle theta.
        """

        dx, dy = self - axis
        cos_t, sin_t = cos(theta), sin(theta)
        return Vec2(
            dx * cos_t - dy * sin_t + axis[0],
            dx * sin_t + dy * cos_t + axis[1]
        )

    def rotated_by(self, theta):
        """
        Return a rotated vector by an angle theta around origin.
        """

        if not isinstance(theta, (float, int)):
            return theta * self

        x, y = self
        cos_t, sin_t = cos(theta), sin(theta)
        return Vec2(x * cos_t - y * sin_t, x * sin_t + y * cos_t)

    def cross(self, other):
        """
        The z component of the cross product between two bidimensional
        smallvectors.
        """

        x, y = other
        return self._x * y - self._y * x

    def polar(self):
        """
        Return a tuple with the (radius, theta) polar coordinates.
        """

        return self.norm(), atan2(self.y, self.x)

    def perpendicular(self, ccw=True):
        """
        Return the counterclockwise perpendicular vector.

        If ccw is False, do the rotation in the clockwise direction.
        """

        if ccw:
            return Vec2(-self.y, self.x)
        else:
            return Vec2(self.y, -self.x)

    #
    # Performance overrides
    #
    def dot(self, other):
        x, y = other
        return self._x * x + self._y * y

    def distance(self, other):
        x, y = other
        x -= self._x
        y -= self._y
        return sqrt(x * x + y * y)

    def angle(self, other):
        cos_t = self.dot(other)
        sin_t = self.cross(other)
        return atan2(sin_t, cos_t)

    def is_null(self, tol=0.0):
        if self._x == 0.0 and self._y == 0.0:
            return True
        elif tol == 0.0:
            return False
        else:
            return super().is_null(tol)

    def is_unity(self, norm=None, tol=1e-6):
        if norm in L2_NORMS:
            return abs(self._x * self._x + self._y * self._y - 1) < 2 * tol
        else:
            return super().is_unity(norm, tol)

    def norm(self, norm=None):
        if norm in L2_NORMS:
            return sqrt(self._x ** 2 + self._y ** 2)
        else:
            return super().norm(norm)

    def norm_sqr(self, norm=None):
        if norm in L2_NORMS:
            return self._x ** 2 + self._y ** 2
        else:
            value = self.norm(norm)
            return value * value

    def normalized(self, norm=None):
        norm = self.norm(norm)
        return Vec2(self._x / norm, self._y / norm)
