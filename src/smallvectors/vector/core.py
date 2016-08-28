from smallvectors.core import Immutable, Mutable, Normed, AddElementWise
from smallvectors.vector.linear import LinearAny

__all__ = ['mVec', 'Vec', 'asvector', 'asmvector', 'asavector']


class VecAny(LinearAny, Normed):
    """
    Base class for Vec and mVec
    """

    def angle(self, other):
        """
        Angle between two vectors.
        """

        try:
            Z = other.norm()
        except AttributeError:
            other = self.vector_type(other)
            Z = other.norm()

        cos_t = self.dot(other) / (self.norm() * Z)
        return self._acos(cos_t)

    def reflect(self, direction):
        """
        Reflect vector around the given direction.
        """

        return self - 2 * (self - self.project(direction))

    def projection(self, direction):
        """
        Returns the projection vector in the given direction
        """

        direction = self.to_direction(direction)
        return self.dot(direction) * direction

    def clamp(self, min_length, max_length=None):
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

    def norm(self, which=None):
        """
        Returns the norm of a vector
        """

        if which is None or which == 'euclidean':
            return self._sqrt(self.norm_sqr())
        elif which == 'minkowski':
            return max(map(abs, self))
        elif which == 'max':
            return max(map(abs, self))
        elif which == 'minkowski':
            return sum(map(abs, self))
        else:
            super().norm(which)

    def norm_sqr(self, which=None):
        """
        Returns the squared norm of a vector.
        """

        if which is None:
            return sum(x * x for x in self)
        else:
            super().norm_sqr(which)

    def rotate(self, rotation):
        """
        Rotate vector by the given rotation object.
        
        The rotation can be specified in several ways which are dimension 
        dependent. All vectors can be irotate by rotation matrices in any
        dimension. 
    
        In 2D, rotation can be a simple number that specifies the 
        angle of rotation.   
        """

        if isinstance(rotation, self._rotmatrix):
            return rotation * self

        tname = type(self).__name__
        msg = 'invalid rotation object for %s: %r' % (tname, rotation)
        raise TypeError(msg)


#
# User-facing types
#
class Vec(VecAny, Immutable):
    """
    Base class for all immutable vector types. Each dimension and type have
    its own related class.
    """

    __slots__ = ()


class mVec(VecAny, Mutable):
    """
    A mutable vector.
    """

    __slots__ = ()

    def irotate(self, rotation):
        """Similar to obj.rotate(...), but make changes *INPLACE*"""

        value = self.rotate(rotation)
        self[:] = value

    def imove(self, *args):
        """
        Similar to obj.move(...), but make changes *INPLACE*
        """

        if len(args) == 1:
            args = args[0]
        self += args

    def ireflect(self, direction):
        """
        Reflect vector around direction *INPLACE*.
        """

        raise NotImplementedError

    def iclamp(self, min_length, max_length=None):
        """
        Like :method:`clamp`, but make changes *INPLACE*.
        """

        if max_length is None:
            ratio = min_length / self.norm()
            self *= ratio

        norm = new_norm = self.norm()
        if norm > max_length:
            new_norm = max_length
        elif norm < min_length:
            new_norm = min_length

        if new_norm != norm:
            self *= new_norm / norm


# Helper functions
def _assure_mutable_set_coord(obj):
    if isinstance(obj, Immutable):
        raise AttributeError('cannot set coordinate of immutable object')


# Vector conversions
def asvector(obj):
    """Return object as an immutable vector"""

    if isinstance(obj, Vec):
        return obj
    else:
        return Vec(*obj)


def asmvector(obj):
    """Return object as a mutable vector"""

    if isinstance(obj, mVec):
        return obj
    else:
        return mVec(*obj)


def asavector(obj):
    """Return object as a mutable or immutable vector.
    
    Non-Vec objects are converted to immutable vectors."""

    if isinstance(obj, LinearAny):
        return obj
    else:
        return Vec(*obj)
