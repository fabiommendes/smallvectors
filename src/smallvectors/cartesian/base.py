# -*- coding: utf8 -*-
'''
Module that implements Vec, Point and Direction base types
'''

import abc
import math
import numbers
import six
from smallvectors.core import VecOrPointMeta, Flat
from smallvectors.generics import add, sub, mul, div, ArithmeticGeneric
from smallvectors.generics import (promote, promote_list, convert_list,
                                   set_promotion_function)

VALID_ELEMENT_TYPES = set()
FORBIDDEN_ELTYPES = {list, tuple, str}

#
# Some aliases
#
make_new = object.__new__
flat_from_list = Flat.from_list_unsafe


class VecOrPointMixin(ArithmeticGeneric):

    '''Common implementations for Vec and Point types'''

    __slots__ = []

    #
    # Overridable math functions. Can be useful when implementing types that
    # uses non-standard numeric types such as decimals or sympy values.
    #
    _sqrt = math.sqrt
    _sin = math.sin
    _cos = math.cos
    _tan = math.tan
    _acos = math.acos
    _asin = math.asin
    _atan = math.atan
    _atan2 = math.atan2
    _number = numbers.Number
    _fast_number = (float, int)

    def __new__(cls, *args, dtype=None):
        if cls.is_root:
            # Prevent creation of zero-dimensional vectors
            if len(args) == 0:
                raise ValueError('no components were given! '
                                 'zero dimensional objects are not allowed')

            # Find the correct type
            if dtype is not None:
                vec_t = cls[len(args), dtype]
            else:
                try:
                    args = promote_list(args)
                except (TypeError, ValueError):
                    vec_t = cls[len(args), object]
                else:
                    vec_t = cls[len(args), type(args[0])]

            return vec_t(*args)

        elif dtype is None or dtype is cls.dtype:
            data = convert_list(args, cls.dtype)
            return cls.from_flat_list_unsafe(data)

        else:
            if not isinstance(dtype, type):
                raise TypeError('dtype must be a type')
            fmt = cls.full_name, dtype.__name__
            raise TypeError('%s cannot set dtype to %s' % fmt)

    #
    # Class constructors
    #
    @classmethod
    @abc.abstractmethod
    def from_flat(cls, data, dtype=None):
        '''Initializes from flattened data. For vectors and points, this is
        equivalent as obj.from_seq(data)'''

        if cls.is_root:
            return cls[len(data), dtype].from_flat(data)
        elif len(data) == cls.size:
            new = object.__new__(cls)
            new.flat = data
            return new
        else:
            fmt = cls.shape[0], data
            msg = 'wrong number of dimensions: expect %sD, got %r' % fmt
            raise ValueError(msg)

    @classmethod
    @abc.abstractmethod
    def from_flat_list_unsafe(cls, data):
        '''Initializes from a list of flattened data of the correct type. This
        is not checked for performance reasons. The user should only use this
        function if it can assure this property for the input data'''

        new = make_new(cls)
        new.flat = flat_from_list(data)
        return new

    @classmethod
    def from_seq(cls, data, dtype=None):
        '''Initializes a Vec[N] from any sequence of N-values'''

        return cls.from_flat(data, dtype=None)

    @classmethod
    def from_coords(cls, *args, dtype=None):
        '''Initializes from coordinates.'''

        return cls.from_flat(args, dtype)

    #
    # Representations
    #
    def as_tuple(self):
        '''Return a tuple of coordinates.'''

        return tuple(self)

    def as_vector(self):
        '''Returns a copy of object as a vector'''

        return self.vector_type(*self)

    def as_direction(self):
        '''Returns a normalized copy of object as a direction'''

        return self.direction_type(*self)

    def as_point(self):
        '''Returns a copy of object as a point'''

        return self.point_type(self)

    def convert(self, dtype):
        '''Convert object to the given data type'''

        cls = type(self)
        if dtype is self.dtype:
            return self
        else:
            return cls.root(*self, dtype=dtype)

    #
    # Geometric properties
    #
    def almost_equals(self, other, tol=1e-3):
        '''Return True if two smallvectors are almost equal to each other'''

        return (self - other).norm_sqr() < tol * tol

    def distance(self, other):
        '''Computes the distance between two objects'''

        if len(self) != len(other):
            N, M = len(self), len(other)
            raise IndexError('invalid dimensions: %s and %s' % (N, M))

        deltas = (x - y for (x, y) in zip(self, other))
        return self._sqrt(sum(x * x for x in deltas))

    def lerp(self, other, weight):
        '''Linear interpolation between two objects.

        The weight attribute defines how close the result will be from the
        argument. A weight of 0.5 corresponds to the middle point between
        the two objects.'''

        if not 0 <= weight <= 1:
            raise ValueError('weight must be between 0 and 1')

        return (other - self) * weight + self

    def middle(self, other):
        '''The midpoint to `other`. The same as ``obj.lerp(other, 0.5)``
        '''

        return (self + other) / 2

    def copy(self, x=None, y=None, z=None, w=None, **kwds):
        '''Return a copy possibly overriding some components'''

        data = list(self)
        if x is not None:
            data[0] = x
        if y is not None:
            data[1] = y
        if z is not None:
            data[2] = z
        if w is not None:
            data[3] = w

        # Keywords of the form e1=foo, e2=bar, etc
        for k, v in kwds.items():
            if not k.startswith('e'):
                raise TypeError('invalid argument %r' % k)
            data[int(k[1:]) + 1] = v

        return self.from_flat(data)

    #
    # Magic methods
    #
    def __repr__(self):
        '''x.__repr__() <==> repr(x)'''

        data = ['%.1f' % x if x != int(x) else str(int(x)) for x in self]
        data = ', '.join(data)
        name = type(self).__root__.__name__
        return '%s(%s)' % (name, data)

    def __str__(self):
        '''x.__str__() <==> str(x)'''

        return repr(self)

    def __hash__(self):
        return hash(self.as_tuple())

    #
    # Abstract methods
    #
    @abc.abstractmethod
    def __len__(self):
        return len(self.flat)

    @abc.abstractmethod
    def __iter__(self):
        return iter(self.flat)

    @abc.abstractmethod
    def __getitem__(self, idx):
        return self.flat[idx]


###############################################################################
#                               Vector types
###############################################################################
class AnyVecMixin(VecOrPointMixin):

    '''Base class for Vec and mVec types'''

    __slots__ = []

    #
    # Constructors
    #
    @classmethod
    def null(cls, dims=None):
        '''Returns the null vector'''

        if dims is None:
            try:
                return cls._null_vector
            except AttributeError:
                data = [0] * cls.dims()
                null_vector = cls._null = cls.from_seq(data)
                return null_vector
        else:
            return cls[dims, cls.dtype].null()

    #
    # Geometric properties
    #
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

    def clamp(self, min, max):
        '''Returns a new vector in which min <= abs(out) <= max.'''

        norm = new_norm = self.norm()
        if norm > max:
            new_norm = max
        elif norm < min:
            new_norm = min

        if new_norm != norm:
            return self.normalize() * new_norm
        else:
            return self

    def dot(self, other):
        '''Dot product between two smallvectors'''

        if len(self) != len(other):
            N, M = len(self), len(other)
            raise ValueError('dimension mismatch: %s and %s' % (N, M))
        return sum(x * y for (x, y) in zip(self, other))

    def is_null(self):
        '''Checks if vector has only null components'''

        return all(x == 0.0 for x in self)

    def is_unity(self, tol=1e-6):
        '''Return True if the norm equals one within the given tolerance'''

        return abs(self.norm() - 1) < tol

    def norm(self):
        '''Returns the norm of a vector'''

        return self._sqrt(self.norm_sqr())

    def norm_sqr(self):
        '''Returns the squared norm of a vector'''

        return sum(x * x for x in self)

    def normalize(self):
        '''Return a normalized version of vector'''

        Z = self.norm()
        return (self / Z if Z != 0.0 else self)

    #
    # Arithmetic operations
    #
    def __mul__(self, other):
        if isinstance(other, self._number):
            dtype = self.dtype
            otype = type(other)
            data = [x * other for x in self]
            if dtype is otype:
                return self.from_flat_list_unsafe(data)
            else:
                return Vec.from_flat(data)

        return mul(self, other)

    def __rmul__(self, other):
        if isinstance(other, self._number):
            dtype = self.dtype
            otype = type(other)
            data = [x * other for x in self]
            if dtype is otype:
                return self.from_flat_list_unsafe(data)
            else:
                return Vec.from_flat(data)

        return mul(other, self)

    def __div__(self, other):
        dtype = self.dtype
        otype = type(other)
        data = [x / other for x in self]
        if dtype is otype:
            return self.from_flat_list_unsafe(data)
        else:
            return mul(self, other)

    __truediv__ = __div__

    def __add__(self, other):
        T_self = type(self)
        T_other = type(other)

        # Two operands of the same or compatible (subclass) types
        if (T_self is T_other) or issubclass(T_other, T_self):
            data = [x + y for (x, y) in zip(self, other)]
            return T_self.from_flat_list_unsafe(data)

        # Two vector types of different dimensions or different element types
        elif isinstance(other, Vec):
            if self.shape != other.shape:
                N, M = self.shape[0], other.shape[0]
                raise TypeError('operation only defined for vectors of same '
                                'dimensions: got %sD vs %sD' % (N, M))

            A, B = promote(self, other)
            return A + B

        # Dispatch to other combinations of types
        else:
            return add(self, other)

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        T_self = type(self)
        T_other = type(other)

        # Two operands of the same or compatible (subclass) types
        if (T_self is T_other) or issubclass(T_other, T_self):
            data = [x - y for (x, y) in zip(self, other)]
            return T_self.from_flat_list_unsafe(data)

        # Two vector types of different dimensions or different element types
        elif isinstance(other, Vec):
            if self.shape != other.shape:
                N, M = self.shape[0], other.shape[0]
                raise TypeError('operation only defined for vectors of same '
                                'dimensions: got %sD vs %sD' % (N, M))

            A, B = promote(self, other)
            return A - B

        # Dispatch to other combinations of types
        else:
            return sub(self, other)

    def __rsub__(self, other):
        # Two vector types of different dimensions or different element types
        if isinstance(other, Vec):
            if self.shape != other.shape:
                N, M = self.shape[0], other.shape[0]
                raise TypeError('operation only defined for vectors of same '
                                'dimensions: got %sD vs %sD' % (N, M))

            A, B = promote(self, other)
            return B - A

        # Dispatch to other combinations of types
        else:
            return sub(other, self)

    def __neg__(self):
        return self.from_flat_list_unsafe([-x for x in self])

    def __nonzero__(self):
        return True

    def __abs__(self):
        return self.norm()

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return all(x == y for (x, y) in zip(self, other))
        elif isinstance(other, (tuple, list, type(self).__root__)):
            match_len = len(self) == len(other)
            match_values = all(x == y for (x, y) in zip(self, other))
            return match_len and match_values
        else:
            return False

    ###########################################################################
    # FIXME: 2D specific functions that should go away
    def rotate(self, theta):
        '''Rotate vector by an angle theta around origin'''

        x, y = self
        cos_t, sin_t = self._cos(theta), self._sin(theta)
        return self.from_coords(
            x * cos_t - y * sin_t,
            x * sin_t + y * cos_t)

    def cross(self, other):
        '''The z component of the cross product between two bidimensional
        smallvectors'''

        x, y = other
        return self.x * y - self.y * x

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]


@six.add_metaclass(VecOrPointMeta)
class Vec(AnyVecMixin):

    '''Base class for all immutable vector types. Each dimension and type have
    its own related class. '''


class Direction(Vec):

    '''Direction is an immutable Vec with unitary length and represents a
    direction in euclidian space'''

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


@six.add_metaclass(VecOrPointMeta)
class mVec(AnyVecMixin):

    '''A mutable Vec'''


###############################################################################
# Point types
###############################################################################
class AnyPoint(VecOrPointMixin):
    pass


@six.add_metaclass(VecOrPointMeta)
class Point(AnyPoint):

    '''Point specific overrides'''

    def __add__(self, other):
        self._assure_match(other)
        if isinstance(other, Point):
            raise TypeError('cannot add two points together')
        return self.from_flat([x + y for (x, y) in zip(self, other)])

    def __radd__(self, other):
        self._assure_match(other)
        return self.from_flat([x + y for (x, y) in zip(self, other)])

    def __sub__(self, other):
        self._assure_match(other)
        data = [x - y for (x, y) in zip(self, other)]
        return self.to_vector(data)

    def __rsub__(self, other):
        self._assure_match(other)
        return self.to_vector([x - y for (x, y) in zip(other, self)])


@six.add_metaclass(VecOrPointMeta)
class mPoint(AnyPoint):

    '''A mutable point type'''


@set_promotion_function(Vec, Vec)
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


@add.overload((Vec, tuple))
@add.overload((Vec, list))
def add_tuple(u, v):
    return u + Vec(*v)


@add.overload((tuple, Vec))
@add.overload((list, Vec))
def radd_tuple(u, v):
    return add_tuple(v, u)


@sub.overload((Vec, tuple))
@sub.overload((Vec, list))
def sub_tuple(u, v):
    return u - Vec(*v)


@sub.overload((tuple, Vec))
@sub.overload((list, Vec))
def rsub_tuple(u, v):
    return add_tuple(u + (-v))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    #
    # Classe
    #
    assert isinstance(Vec, VecOrPointMeta)
    assert Vec[2, float] is Vec[2, float]
    assert Vec[2].shape == (2,)

    # Comparações
    u = Vec(1, 2)
    v = Vec(3, 4)
    w = Vec(1.0, 2)
    u_type = type(u)
    v_type = type(v)
    sum_type = type(u + v)
    assert u.dtype is int
    assert w.dtype is float
    assert (u + w).dtype is float, (u + w).dtype
    assert Vec(1, 2) == Vec(1, 2)
    assert Vec(1, 2, dtype=float) == Vec(1, 2, dtype=int)
    assert u_type is v_type
    assert u_type is v_type is sum_type

    # Mathematical operations
    assert u + v == Vec(4, 6)
    assert u - v == Vec(-2, -2)
    assert 2 * u == Vec(2, 4)
    assert 2.0 * u == Vec(2.0, 4.0)

    # Operations with tuples and other objects
    assert u == (1, 2)
    assert u + (1, 2) == (2, 4)
    assert u * 1 == u
    assert 1 * u == u

    u3 = Vec(1, 2, 3)

    import time
    print(type(u), type(v), type(u + v))
    u_list = [u] * 100000
    t0 = time.time()
    sum(u_list, v)
    print(time.time() - t0)

    print(Vec.mro())
    print(Vec[2, int].mro())