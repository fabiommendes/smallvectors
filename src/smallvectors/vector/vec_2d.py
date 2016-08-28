from generic import convert, InexactError, get_conversion

from smallvectors.core import Mutable
from smallvectors.vector.vec import mVec, Vec, _assure_mutable_set_coord
from smallvectors.vector.vec_nd import VecND, Vec1D


class Vec2D(VecND):
    """
    Vector functions that only works in 2D.

    These functions are inserted to all Vec[2, ...] classes upon class
    creation.
    """

    __slots__ = ('_x', '_y')

    def __init__(self, x, y):
        self._x = convert(x, self.dtype)
        self._y = convert(y, self.dtype)

    def __len__(self):
        return 2

    def __iter__(self):
        yield self._x
        yield self._y

    def __getitem_simple__(self, idx):
        if idx == 0:
            return self._x
        elif idx == 1:
            return self._y
        else:
            raise RuntimeError('invalid index for getitem_simple: %s' % idx)

    def __addsame__(self, other):
        return self._fromcoords_unsafe(self._x + other._x, self._y + other._y)

    def __subsame__(self, other):
        return self._fromcoords_unsafe(self._x - other._x, self._y - other._y)

    def __mul__(self, other):
        x = self._x * other
        y = self._y * other
        if isinstance(x, self.dtype):
            return self._fromcoords_unsafe(x, y)
        elif isinstance(other, self._number):
            return self.__origin__(x, y)
        else:
            return NotImplemented

    def __truediv__(self, other):
        return self * (1.0 / other)

    #
    # Constructors
    #
    @classmethod
    def fromflat(cls, data, copy=True):
        x, y = data
        return cls._fromcoords_unsafe(x, y)

    @classmethod
    def frompolar(cls, radius, theta=0):
        """Create vector from polar coordinates"""
        return cls(radius * cls._cos(theta), radius * cls._sin(theta))

    @classmethod
    def _fromcoords_unsafe(cls, x, y):
        new = object.__new__(cls)
        new._x = x
        new._y = y
        return new

    #
    # Properties
    #
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        _assure_mutable_set_coord(self)
        self._y = convert(value, self.dtype)

    x = x0 = Vec1D.x
    x1 = y

    #
    # 2D specific API
    #
    def rotated_axis(self, axis, theta):
        """
        Rotate vector around given axis by the angle theta.
        """

        dx, dy = self - axis
        cos_t, sin_t = self._cos(theta), self._sin(theta)
        return self._fromcoords_unsafe(
            dx * cos_t - dy * sin_t + axis[0],
            dx * sin_t + dy * cos_t + axis[1])

    def rotate(self, theta):
        """
        Rotate vector by an angle theta around origin.
        """

        if isinstance(theta, self._rotmatrix):
            return theta * self

        cls = type(self)
        x, y = self
        cos_t, sin_t = self._cos(theta), self._sin(theta)

        # TODO: decent implementation of this!
        try:
            return cls(x * cos_t - y * sin_t, x * sin_t + y * cos_t)
        except InexactError:
            if isinstance(self, Mutable):
                return mVec(x * cos_t - y * sin_t, x * sin_t + y * cos_t)
            else:
                return Vec(x * cos_t - y * sin_t, x * sin_t + y * cos_t)

    def cross(self, other):
        """
        The z component of the cross product between two bidimensional
        smallvectors.
        """

        x, y = other
        return self.x * y - self.y * x

    def polar(self):
        """
        Return a tuple with the (radius, theta) polar coordinates.
        """

        return (self.norm(), self._atan2(self.y, self.x))

    def perp(self, ccw=True):
        """
        Return the counterclockwise perpendicular vector.

        If ccw is False, do the rotation in the clockwise direction.
        """

        if ccw:
            return self._fromcoords_unsafe(-self.y, self.x)
        else:
            return self._fromcoords_unsafe(self.y, -self.x)

    #
    # Performance overrides
    #
    def distance(self, other):
        return self._sqrt((other.x - self._x) ** 2 + (other.y - self._y) ** 2)

    def convert(self, dtype):
        _float = float

        if dtype is self.dtype:
            return self
        elif dtype is _float:
            x = _float(self._x)
            y = _float(self._y)

            try:
                return self.__float_root(x, y)
            except AttributeError:
                self_t = type(self)
                cls = self_t.__float_root = self.__origin__[2, float]
                return cls(x, y)
        else:
            conv = get_conversion(self.dtype, dtype)
            x = conv(self._x)
            y = conv(self._y)
            return self.__origin__[2, dtype](x, y)

    def angle(self, other):
        """Computes the angle between two smallvectors"""

        cos_t = self.dot(other)
        sin_t = self.cross(other)
        return self._atan2(sin_t, cos_t)

    def is_null(self):
        """Checks if vector has only null components"""

        if self._x == 0.0 and self._y == 0.0:
            return True
        else:
            return False

    def is_unity(self, norm=None, tol=1e-6):
        """
        Return True if the norm equals one within the given tolerance.
        """

        if norm is None:
            return abs(self._x * self._x + self._y * self._y - 1) < 2 * tol
        else:
            return super().is_unity(norm, tol)

    def norm(self, which=None):
        """
        Returns the norm of a vector.
        """

        if which is None:
            return self._sqrt(self._x ** 2 + self._y ** 2)
        else:
            return Vec.norm(self, which)

    def norm_sqr(self, which=None):
        """
        Returns the squared norm of a vector.
        """

        if which is None:
            return self._x ** 2 + self._y ** 2
        else:
            return Vec.norm(self, which)