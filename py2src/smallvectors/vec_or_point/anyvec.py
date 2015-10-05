from __future__ import division
from generic import promote, set_promotion
from generic.operator import add, sub
from ..core import Immutable, Mutable, conditional_method
from . import VecOrPoint


class AnyVec(VecOrPoint):

    '''Base class for Vec and mVec types'''

    __slots__ = ()

    def angle(self, other):
        '''Angle between two vectors'''

        try:
            Z = other.norm()
        except AttributeError:
            other = self.vector_type(other)
            Z = other.norm()

        cos_t = self.dot(other) / (self.norm() * Z)
        return self._acos(cos_t)

    def reflect(self, direction):
        '''Reflect vector around the given direction'''

        return self - 2 * (self - self.project(direction))

    def project(self, direction):
        '''Returns the projection vector in the given direction'''

        direction = self.to_direction(direction)
        return self.dot(direction) * direction

    def clamp(self, min_length, max_length):
        '''Returns a new vector in which min <= abs(out) <= max.'''

        norm = new_norm = self.norm()
        if norm > max_length:
            new_norm = max_length
        elif norm < min_length:
            new_norm = min_length

        if new_norm != norm:
            return self.normalize() * new_norm
        else:
            return self

    def dot(self, other):
        '''Dot product between two objects'''

        if len(self) != len(other):
            N, M = len(self), len(other)
            raise ValueError('dimension mismatch: %s and %s' % (N, M))
        return sum(x * y for (x, y) in zip(self, other))

    def norm(self, which=None):
        '''Returns the norm of a vector'''

        # TODO: different norms: euclidean, max, minkoski, etc
        return self._sqrt(self.norm_sqr())

    def norm_sqr(self, which=None):
        '''Returns the squared norm of a vector'''

        if which is None:
            return sum(x * x for x in self)
        else:
            super(AnyVec, self).norm_sqr(which)

    #
    # 2D specific functions
    #
    @classmethod
    @conditional_method(lambda p: p[0] == 2)
    def from_polar(cls, radius, theta=0):
        '''Create vector from polar coordinates'''

        return cls(radius * cls._cos(theta), radius * cls._sin(theta))

    @conditional_method(lambda p: p[0] == 2)
    def rotate(self, theta):
        '''Rotate vector by an angle theta around origin'''

        cls = type(self)
        x, y = self
        cos_t, sin_t = self._cos(theta), self._sin(theta)
        return cls(x * cos_t - y * sin_t, x * sin_t + y * cos_t)

    @conditional_method(lambda p: p[0] == 2)
    def cross(self, other):
        '''The z component of the cross product between two bidimensional
        smallvectors'''

        x, y = other
        return self.x * y - self.y * x


class Vec(AnyVec, Immutable):

    '''Base class for all immutable vector types. Each dimension and type have
    its own related class. '''

    __slots__ = ()


class mVec(AnyVec, Mutable):

    '''A mutable vector'''

    __slots__ = ()


class Direction(Vec):

    '''Direction is an immutable Vec with unitary length and represents a
    direction in euclidian space'''

    __slots__ = ()

    def is_null(self):
        '''Always False for Direction objects'''

        return False

    def is_unity(self, tol=1e-6):
        '''Always True for Direction objects'''

        return True

    def norm(self):
        '''Returns the norm of a vector'''

        return 1.0

    def norm_sqr(self):
        '''Returns the squared norm of a vector'''

        return 1.0

    def normalize(self):
        '''Return a normalized version of vector'''

        return self


@set_promotion(Vec, Vec)
def promote_vectors(u, v):
    '''Promote two Vec types to the same type'''

    u_type = type(u)
    v_type = type(v)
    if u_type is v_type:
        return (u, v)

    # Test shapes
    if u_type.shape != v_type.shape:
        raise TypeError('vectors have different shapes')

    # Fasttrack common cases
    u_dtype = u.dtype
    v_dtype = v.dtype
    if u_dtype is float and v_dtype is int:
        return u, v.convert(float)
    elif u_dtype is int and v_dtype is float:
        return u.convert(float), v

    zipped = [promote(x, y) for (x, y) in zip(u, v)]
    u = Vec(*[x for (x, y) in zipped])
    v = Vec(*[y for (x, y) in zipped])
    return u, v

if __name__ == '__main__':
    def test_consistent_class_parameters():
        assert Vec[2, int].dtype is int
        assert Vec[2, int].shape == (2,)
        assert Vec[2, int].size == 2
        assert Vec[2, int].ndim == 1
        assert Vec[2, int].__root__ is Vec, [Vec[2, int].__root_, Vec]

    test_consistent_class_parameters()

    u = Vec(1, 2)
    print(u)
    Vec2 = Vec[2, float]
    print(Vec2)
    v = Vec2(1, 2)
    print(v)
    print(v + v)
    print(u + v)
    print(u * 2)
    print(isinstance(v, Vec), isinstance(v, AnyVec), isinstance(v, mVec))

    import time
    print(type(v), type(v), type(u + v))
    u_list = [v] * 100000
    t0 = time.time()
    sum(u_list, v)
    print(time.time() - t0)
