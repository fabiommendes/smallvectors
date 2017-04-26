import abc
from math import sqrt, acos, pi

from .linear import Linear
from ..interfaces import Normed

L1_NORMS = ('L1', 'l1', 'manhattan')
L2_NORMS = (None, 'L2', 'l2', 'euclidean')


class Vec(Linear, Normed, metaclass=abc.ABCMeta):
    """
    Base class for all VecN classes.
    """

    __slots__ = ()
    _rotmat_class = None
    ndim = 1
    dtype = float

    def __repr__(self):
        data = ', '.join('%g' % x for x in self)
        return '%s(%s)' % (self.__class__.__name__, data)

    def __neg__(self):
        return self * (-1)

    def __add__(self, other):
        cls = type(self)
        if isinstance(other, (cls, list, tuple)):
            if len(other) != len(self):
                raise ValueError('dimensions do not match')
            return cls(*(x + y for (x, y) in zip(self, other)))
        return NotImplemented

    def __sub__(self, other):
        cls = type(self)
        if isinstance(other, (cls, list, tuple)):
            if len(other) != len(self):
                raise ValueError('dimensions do not match')
            return cls(*(x - y for (x, y) in zip(self, other)))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            cls = type(self)
            return cls(*(x * other for x in self))
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            cls = type(self)
            return cls(*(x // other for x in self))
        return NotImplemented

    def __truediv__(self, other):
        return self * (1 / other)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __abs__(self):
        return self.norm()

    def as_vector(self):
        return self

    def angle(self, other):
        """
        Angle between two vectors.
        """
        from .functions import asvector

        try:
            Z = other.norm()
        except AttributeError:
            other = asvector(other)
            Z = other.norm()

        cos_t = self.dot(other) / (self.norm() * Z)
        if cos_t >= 1.0:
            return 0.0
        elif cos_t <= -1.0:
            return pi
        return acos(cos_t)

    def reflexion(self, direction):
        """
        Return reflexion of vector over the given direction.
        """

        return self - 2 * (self - self.projection(direction))

    def projection(self, direction):
        """
        Returns the projection vector in the given direction
        """

        direction = self.as_direction()
        return self.dot(direction) * direction

    def clamped(self, min_length, max_length=None):
        """
        Returns a new vector in which min_length <= abs(out) <=
        max_length.
        """

        if max_length is None:
            ratio = min_length / self.norm()
            return self * ratio

        norm = new_norm = self.norm()
        if norm > max_length:
            new_norm = max_length
        elif norm < min_length:
            new_norm = min_length

        if new_norm != norm:
            return self.normalized() * new_norm
        else:
            return self.copy()

    def dot(self, other):
        """
        Dot product between two objects.
        """

        if len(self) != len(other):
            N, M = len(self), len(other)
            raise ValueError('dimension mismatch: %s and %s' % (N, M))
        return sum(x * y for (x, y) in zip(self, other))

    def norm(self, norm=None):
        """
        Return the vector's norm.
        """

        if norm in L2_NORMS:
            return sqrt(sum(x * x for x in self))
        elif norm in L1_NORMS:
            return sum(map(abs, self))
        elif norm == 'max':
            return max(map(abs, self))
        else:
            raise ValueError('invalid norm: %r' % norm)

    def norm_sqr(self, norm=None):
        """
        Return the squared norm.
        """

        if norm in L2_NORMS:
            return sum(x * x for x in self)
        else:
            return super().norm_sqr(norm)

    def rotated_by(self, rotation):
        """
        Rotate vector by the given rotation object.
        
        The rotation can be specified in several ways which are dimension 
        dependent. All vectors can be rotated by rotation matrices in any
        dimension. 
    
        In 2D, rotation can be a simple number that specifies the 
        angle of rotation.   
        """

        if isinstance(rotation, self._rotmat_class):
            return rotation * self

        tname = type(self).__name__
        msg = 'invalid rotation object for %s: %r' % (tname, rotation)
        raise TypeError(msg)
