from generic import convert

from .vec import _assure_mutable_set_coord
from .vec_3d import Vec3D
from .vec_nd import VecND


class Vec4D(VecND):
    """
    Vector functions that only works in 4D.

    These functions are inserted to all Vec[4, ...] classes upon class
    creation.
    """

    __slots__ = ('_x', '_y', '_z', '_w')

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        _assure_mutable_set_coord(self)
        self._w = value

    x = x0 = Vec3D.x
    y = x1 = Vec3D.y
    z = x2 = Vec3D.z
    x3 = w

    @classmethod
    def from_flat(cls, data, copy=True, dtype=None, shape=None):
        cls._check_params(shape, dtype)
        x, y, z, w = data
        return cls._from_coords_unsafe(x, y, z, w)

    @classmethod
    def _from_coords_unsafe(cls, x, y, z, w):
        new = object.__new__(cls)
        new._x = x
        new._y = y
        new._z = z
        new._w = w
        return new

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

    def __getitem_simple__(self, idx):
        if idx == 0:
            return self._x
        elif idx == 1:
            return self._y
        elif idx == 2:
            return self._z
        elif idx == 3:
            return self._w

    def __setitem__(self, idx, value):
        _assure_mutable_set_coord(self)
        value = convert(value, self.dtype)
        if idx == 0:
            self._x = value
        elif idx == 1:
            self._y = value
        elif idx == 2:
            self._z = value
        elif idx == 3:
            self._w = value
