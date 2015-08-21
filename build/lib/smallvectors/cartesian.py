# -*- coding: utf8 -*-
'''
Module that implements Vec, Point and Direction base types
'''

import abc
import math
import numbers
import six
from smallvectors.metatypes import VecOrPointMeta

PROMOTION_RULES = {}
VALID_ELEMENT_TYPES = set()
FORBIDDEN_ELTYPES = {list, tuple, str}
CONVERT_RULES = {}


def convert(ret_type, value):
    '''Convert value to the given return type.

    It raises a TypeError if no conversion is possible and a ValueError if
    conversion is possible in general, but not for the specific value given.'''

    try:
        converter = CONVERT_RULES[ret_type, type(value)]
    except KeyError:
        fmt = type(value).__name__, ret_type.__name__
        raise TypeError('no conversion found from %s to %s' % fmt)
    else:
        return converter(value)


def define_conversion(from_type, to_type, function):
    '''Register a function that convert between the two given types'''

    # Forbid redefinitions
    if (from_type, to_type) in CONVERT_RULES:
        fmt = from_type.__name__, to_type.__name__
        raise ValueError('cannot overwrite convertion from %s to %s' % fmt)

    CONVERT_RULES[from_type, to_type] = function


def promote_type(T1, T2):
    '''Promote two types to the type which has highest resolution. Raises a
    TypeError if promotion was not found'''

    # Fast track common types
    if T1 is float and T2 is int:
        return float
    if T2 is float and T1 is int:
        return float

    # Check the promotions dictionary
    try:
        return PROMOTION_RULES[T1, T2]
    except KeyError:
        pass

    # Check if types are the same
    if T1 is T2:
        return T1

    aux = (T1.__name__, T2.__name__)
    raise TypeError('no promotion rule found for %s and %s' % aux)


def promote_value(value, T):
    '''Promote two types to the type which has highest resolution. Raises a
    TypeError if promotion was not found'''

    cls = type(value)

    # Fast track common types
    if cls is float and T is int:
        return value
    if T is float and cls is int:
        return float(value)

    # Check the promotions dictionary
    try:
        return PROMOTION_RULES[cls, T](value)
    except KeyError:
        pass

    # Check if types are the same
    if cls is T:
        return value

    aux = (cls.__name__, T.__name__)
    raise TypeError('no promotion rule found for %s and %s' % aux)


def select_type(values):
    '''Return the best (most generic) type from a list of values'''

    if not values:
        raise ValueError('empty list')

    values = iter(values)
    ret_type = type(next(values))

    for value in values:
        tt = type(value)
        if tt is not ret_type:
            ret_type = promote_type(tt, ret_type)

    assert ret_type in VALID_ELEMENT_TYPES, ret_type
    return ret_type


def define_promotion(T1, T2, T3, converter=None):
    '''Define the promotion rule for the pair (T1, T2)'''

    if not (T3 is T1 or T3 is T2):
        raise ValueError('must promote to either T1 or T2')
    PROMOTION_RULES[T1, T2] = PROMOTION_RULES[T2, T1] = T3

    for T in [T1, T2, T3]:
        VALID_ELEMENT_TYPES.add(T)

define_promotion(float, int, float)
define_promotion(float, float, float)
define_promotion(int, int, int)


class VecOrPointMixin(object):

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
        # Prevent creation of zero-dimensional vectors
        if len(args) == 0:
            raise ValueError('no components were given! '
                             'zero dimensional objects are not allowed')

        if dtype is None:
            if cls.is_root():
                dtype = select_type(args)
                base_class = cls[len(args), dtype or float]
            else:
                return cls.from_flat(args)

        elif cls.is_root():
            base_class = cls[len(args), dtype or float]

        if dtype in FORBIDDEN_ELTYPES:
            clsname = cls.__name__
            elname = dtype.__name__
            raise TypeError('%s elements cannot be %s' % (clsname, elname))

        return base_class.from_flat(args)

    #
    # Class constructors
    #
    @classmethod
    @abc.abstractmethod
    def from_flat(cls, data, dtype=None):
        '''Initializes from flattened data. For vectors and points, this is
        equivalent as obj.from_seq(data)'''

        if cls.is_root():
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
    def _find_common_type(self, other):
        if isinstance(other, tuple):
            return self, self.from_seq(other)
        elif isinstance(other, Vec):
            # FIXME: make it right!
            return self, Vec(*other, dtype=float)
        raise ValueError(other)

    def __mul__(self, other):
        if isinstance(other, self._number):
            dtype = self.dtype
            otype = type(other)
            if dtype is otype:
                return Vec.from_seq([x * other for x in self], dtype=dtype)
            else:
                return Vec.from_seq([x * other for x in self])
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return self.from_seq([x / other for x in self])

    def __truediv__(self, other):
        return self.from_seq([x / other for x in self])

    def __add__(self, other):
        T_self = self.__class__
        T_other = other.__class__

        if T_self is T_other or issubclass(T_other, T_self):
            return self.__add_same__(other)
        else:
            A, B = self._find_common_type(other)
            return A.__add_same__(B)

    def __add_same__(self, other):
        data = [x + y for (x, y) in zip(self, other)]
        return Vec.from_seq(data, dtype=self.dtype)

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        T_self = self.__class__
        T_other = other.__class__

        if T_self is T_other or issubclass(T_other, T_self):
            return self.__sub_same__(other)
        else:
            A, B = self._find_common_type(other)
            return A.__sub_same__(B)

    def __sub_same__(self, other):
        data = [x - y for (x, y) in zip(self, other)]
        return Vec.from_seq(data, dtype=self.dtype)

    def __rsub__(self, other):
        self._assure_match(other)
        return self.to_vector([x - y for (x, y) in zip(other, self)])

    def __neg__(self):
        return self.from_flat([-x for x in self])

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

    '''Base class for all immutable vector types. Each dimension and type'''


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


#
# Late binding
#
# Add Vec, Point and Direction as forbidden element types
FORBIDDEN_ELTYPES.add(Vec)
FORBIDDEN_ELTYPES.add(Point)
FORBIDDEN_ELTYPES.add(Direction)

#
# Classe
#
assert isinstance(Vec, VecOrPointMeta)
assert Vec[2, float] is Vec[2, float]
assert Vec[2].shape == (2,)
# assert Vec(1, 2).dtype is object  # ???

# Comparações
u = Vec(1, 2)
v = Vec(3, 4)
assert Vec(1, 2) == Vec(1, 2)
assert Vec(1, 2, dtype=float) == Vec(1, 2, dtype=int)

# Mathematical operations
assert u + v == Vec(4, 6)
assert u - v == Vec(-2, -2)
assert 2 * u == Vec(2, 4)


u3 = Vec(1, 2, 3)
#u3 + u

if __name__ == '__main__':
    import doctest
    doctest.testmod()
