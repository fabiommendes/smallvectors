import abc

import collections
from math import sqrt


class Linear(collections.Sequence, metaclass=abc.ABCMeta):
    """
    Common implementations for Vec and Point types
    """

    size = None
    _vector_class = None
    _direction_class = None
    _point_class = None
    __slots__ = ()
    flat = property(lambda self: list(self))

    @classmethod
    def from_flat(cls, data):
        return cls(*data)

    def __len__(self):
        return self.size

    def __pos__(self):
        return self

    # Representation and conversions
    def as_vector(self):
        """
        Returns a copy of object as a vector
        """

        return self._vector_class(*self)

    def as_direction(self):
        """
        Returns a normalize copy of object as a direction
        """

        return self._direction_class(*self)

    def as_point(self):
        """
        Returns a copy of object as a point.
        """

        return self._point_class(*self)

    @abc.abstractmethod
    def copy(self, x=None, y=None, z=None, w=None, **kwds):
        """
        Return a copy possibly overriding some components.
        """

        data = list(self)
        if x is not None:
            data[0] = x
        if y is not None:
            data[1] = y
        if z is not None:
            data[2] = z
        if w is not None:
            data[3] = w

        # Keywords of the form x1=foo, x2=bar, etc
        for k, v in kwds.items():
            if not k.startswith('x'):
                raise TypeError('invalid argument %r' % k)
            data[int(k[1:]) + 1] = v

        return self.from_flat(data)

    # Geometric properties
    def almost_equal(self, other, tol=1e-3):
        """
        Return True if two vectors are almost equal to each other.
        """

        return (self - other).norm_sqr() < tol * tol

    def distance(self, other):
        """
        Computes the distance between two objects.
        """

        if len(self) != len(other):
            N, M = len(self), len(other)
            raise IndexError('invalid dimensions: %s and %s' % (N, M))

        deltas = (x - y for (x, y) in zip(self, other))
        return sqrt(sum(x * x for x in deltas))

    def lerp(self, other, weight=0.5):
        """
        Linear interpolation between two objects.

        The weight attribute defines how close the result will be from the
        argument. A weight of 0.5 corresponds to the middle point between
        the two objects.
        """

        if not 0 <= weight <= 1:
            raise ValueError('weight must be between 0 and 1')

        return (other.as_vector() - self) * weight + self

    def middle(self, other):
        """
        The midpoint to `other`. The same as ``obj.lerp(other, 0.5)``.
        """

        return (self + other.as_vector()) / 2

    def displaced_by(self, *args):
        """
        Displace object by the given coordinates.

        This is equivalent to a vector sum.
        """

        if len(args) == 1 and isinstance(args[0], Linear):
            args = args[0]
        return self + args
