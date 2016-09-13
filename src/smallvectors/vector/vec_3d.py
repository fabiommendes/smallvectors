from generic import convert

from .vec import Vec, _assure_mutable_set_coord
from .vec_2d import Vec2D
from .vec_nd import VecND


class Vec3D(VecND):
    """
    Vector functions that only work in 3D.

    These functions are inserted to all Vec[3, ...] classes upon class
    creation.
    """

    __slots__ = ('_x', '_y', '_z')

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        _assure_mutable_set_coord(value)
        self._z = value

    x = x0 = Vec2D.x
    y = x1 = Vec2D.y
    x3 = z

    @classmethod
    def from_flat(cls, data, copy=True, dtype=None, shape=None):
        if shape is not None and shape != (3,):
            raise TypeError('Vec3D cannot have a shape different from (3,)')
        if dtype is None or dtype is cls.dtype:
            x, y, z = data
            return cls._from_coords_unsafe(x, y, z)
        else:
            return cls._from_coords_unsafe(*(convert(x, dtype) for x in data))

    @classmethod
    def from_spherical(cls, radius, phi=0, theta=0):
        """
        Create vector from spherical coordinates.
        """

        r = radius * cls._sin(phi)
        x = r * cls._cos(theta)
        y = r * cls._sin(theta)
        z = r * cls._cos(phi)
        return cls(x, y, z)

    @classmethod
    def from_cylindrical(cls, radius, theta=0, z=0):
        """
        Create vector from cylindrical coordinates.
        """

        x = radius * cls._cos(theta)
        y = radius * cls._sin(theta)
        return cls(x, y, z)

    @classmethod
    def _from_coords_unsafe(cls, x, y, z):
        new = object.__new__(cls)
        new._x = x
        new._y = y
        new._z = z
        return new

    def __init__(self, x, y, z):
        dtype = self.dtype
        self._x = convert(x, dtype)
        self._y = convert(y, dtype)
        self._z = convert(z, dtype)

    def __len__(self):
        return 3

    def __iter__(self):
        yield self._x
        yield self._y
        yield self._z

    def __getitem_simple__(self, idx):
        if idx == 0:
            return self._x
        elif idx == 1:
            return self._y
        elif idx == 2:
            return self._z
        else:
            raise RuntimeError('invalid integer index: %s' % idx)

    def __setitem__(self, idx, value):
        _assure_mutable_set_coord(self)
        value = convert(value, self.dtype)
        if idx == 0:
            self._x = value
        elif idx == 1:
            self._y = value
        elif idx == 2:
            self._z = value

    def cross(self, other):
        """
        The cross product between two tridimensional vectors.
        """

        x, y, z = self
        a, b, c = other
        return Vec(y * c - z * b, z * a - x * c, x * b - y * a)
