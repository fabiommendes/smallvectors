from generic import convert

from .core import _assure_mutable_set_coord
from .vec_3d import Vec3D
from .vec_nd import VecND


class Vec4D(VecND):
    """
    Vector functions that only works in 4D.

    These functions are inserted to all Vec[4, ...] classes upon class
    creation.
    """

    def __init__(self, x, y, z, w):
        dtype = self.dtype
        self._x = convert(x, dtype)
        self._y = convert(y, dtype)
        self._z = convert(z, dtype)
        self._w = convert(w, dtype)

    def __len__(self):
        return 4

    def __iter__(self):
        yield self._x
        yield self._y
        yield self._z
        yield self._w

    def __getitem__(self, idx):
        if idx == 0:
            return self._x
        elif idx == 1:
            return self._y
        elif idx == 2:
            return self._z
        elif idx == 3:
            return self._w
        else:
            raise RuntimeError('invalid index for getitem_simple: %s' % idx)

    @classmethod
    def fromflat(cls, data, copy=True):
        x, y, z, w = data
        return cls._fromcoords_unsafe(x, y, z, w)

    @classmethod
    def fromspheric(cls, radius, phi=0, theta=0):
        """Create vector from spherical coordinates"""

        r = radius * cls._sin(phi)
        x = r * cls._cos(theta)
        y = r * cls._sin(theta)
        z = r * cls._cos(phi)
        return cls(x, y, z)

    @classmethod
    def fromcylindric(cls, radius, theta=0, z=0):
        """Create vector from cylindric coordinates"""

        x = radius * cls._cos(theta)
        y = radius * cls._sin(theta)
        return cls(x, y, z)

    @classmethod
    def _fromcoords_unsafe(cls, x, y, z):
        new = object.__new__(cls)
        new._x = x
        new._y = y
        new._z = z
        return new

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        _assure_mutable_set_coord(value)
        self._w = value

    x = x0 = Vec3D.x
    y = x1 = Vec3D.y
    z = x2 = Vec3D.z
    x3 = w
