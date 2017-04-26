from math import sin, cos, sqrt, atan2

from .vec import Vec


class Vec3(Vec):
    """
    A 3D vector.
    """

    __slots__ = ('_x', '_y', '_z')
    size = 3
    shape = (3,)

    x0 = x = property(lambda self: self._x)
    x1 = y = property(lambda self: self._y)
    x2 = z = property(lambda self: self._z)

    @classmethod
    def from_flat(cls, data):
        x, y, z = data
        return Vec3(x, y, z)

    @classmethod
    def from_spherical(cls, radius, phi=0, theta=0):
        """
        Create vector from spherical coordinates.
        """

        r = radius * sin(phi)
        x = r * cos(theta)
        y = r * sin(theta)
        z = r * cos(phi)
        return cls(x, y, z)

    @classmethod
    def from_cylindrical(cls, radius, theta=0, z=0):
        """
        Create vector from cylindrical coordinates.
        """

        x = radius * cos(theta)
        y = radius * sin(theta)
        return cls(x, y, z)

    def __init__(self, x, y, z):
        self._x = x + 0.0
        self._y = y + 0.0
        self._z = z + 0.0

    def __len__(self):
        return 3

    def __iter__(self):
        yield self._x
        yield self._y
        yield self._z

    def __getitem__(self, idx):
        if idx in (0, -3):
            return self._x
        elif idx in (1, -2):
            return self._y
        elif idx in (2, -1):
            return self._z
        elif isinstance(idx, slice):
            return [self._x, self._y, self._z][idx]
        else:
            raise IndexError(idx)

    def __eq__(self, other):
        if isinstance(other, (Vec3, list, tuple)):
            try:
                x, y, z = other
            except ValueError:
                return False
            return self._x == x and self._y == y and self._z == z
        return NotImplemented

    def copy(self, x=None, y=None, z=None, **kwargs):
        if kwargs:
            return super().copy(x=x, y=y, z=z, **kwargs)
        if x is None:
            x = self._x
        if y is None:
            y = self._y
        if z is None:
            z = self._z
        return Vec3(x, y, z)

    def spherical(self):
        """
        Return rho, phi, theta spherical coordinates, where rho is the vector
        length, phi is the angle with the z-axis and theta is the angle the
        projection in the xy plane makes with the x axis.
        """

        x, y, z = self
        r2 = x * x + y * y
        r = sqrt(r2)
        return sqrt(r2 + z * z), atan2(r, z), atan2(x, y)

    def cylindrical(self):
        """
        Return the r, theta, z cylindrical coordinates, where r is the length
        of the projection in the xy-plane, theta is the angle of this projection
        with the x axis and z is the z coordinate of the vector.
        """

        x, y, z = self
        return sqrt(x * x + y * y), atan2(x, y), z

    def cross(self, other):
        """
        The cross product between two tridimensional vectors.
        """

        x, y, z = self
        a, b, c = other
        return Vec3(y * c - z * b, z * a - x * c, x * b - y * a)
