'''
Example
-------

Criamos um vetor chamando a classe com as componentes como argumento.

>>> v = Vec(3, 4); print(v)
Vec[2, int](3, 4)

Os métodos de listas funcionam para objetos do tipo Vec2:

>>> v[0], v[1], len(v)
(3, 4, 2)

Objetos do tipo Vec2 também aceitam operações matemáticas

>>> v + 2 * v
Vec[2, int](9, 12)

Além de algumas funções de conveniência para calcular o módulo,
vetor unitário, etc.

>>> v.norm(); abs(v)
5.0
5.0

>>> v.normalized().almost_equal(Vec[2, float](0.6, 0.8))
True
'''
from generic import convert, GenericObject, get_conversion
from ..core import Flat
from . import AnyVec, Vec, mVec, Direction
from . import AnyPoint, Point, mPoint


class Base2D(GenericObject):

    '''Base class for Vec2, Direction2 and Point2 classes'''

    __slots__ = ('_x', '_y')

    def __init__(self, x, y):
        self._x = convert(x, self.dtype)
        self._y = convert(y, self.dtype)

    @classmethod
    def _from_coords_unsafe(cls, x, y):
        new = object.__new__(cls)
        new._x = x
        new._y = y
        return new

    #
    # Abstract methods overrides
    #
    @classmethod
    def from_flat(cls, data, copy=True):
        x, y = data
        return cls._from_coords_unsafe(x, y)

    @property
    def flat(self):
        return Flat(self)

    def __len__(self):
        return 2

    def __iter__(self):
        yield self._x
        yield self._y

    def __getitem__(self, i):
        '''x.__getitem__(i) <==> x[i]'''

        if i == 0:
            return self._x
        elif i == 1:
            return self._y
        else:
            raise IndexError(i)

    #
    # Performance overrides
    #
    def distance(self, other):
        return self._sqrt((other.x - self._x) ** 2 + (other.y - self._y) ** 2)

    def convert(self, dtype):
        _float = float

        if dtype is self.dtype:
            return self

        elif dtype is _float:
            x = _float(self._x)
            y = _float(self._y)

            try:
                return self.__float_root(x, y)
            except AttributeError:
                self_t = type(self)
                cls = self_t.__float_root = self.__root__[2, float]
                return cls(x, y)

        else:
            conv = get_conversion(self.dtype, dtype)
            x = conv(self._x)
            y = conv(self._y)
            return self.__root__[2, dtype](x, y)


@Vec.subtype_register_base(shape=(2,))
class VecBase2D(Base2D):

    '''Base class with common implementations for for Vec2 and Direction2'''

    __slots__ = ()

    #
    # Performance overrides
    #

    def angle(self, other):
        '''Computes the angle between two smallvectors'''

        cos_t = self.dot(other)
        sin_t = self.cross(other)
        return self._atan2(sin_t, cos_t)

    #
    # 2D specific geometric properties and operations
    #
    def polar(self):
        '''Return a tuple with the (radius, theta) polar coordinates '''

        return (self.norm(), self._atan2(self._y, self._x))

    def perp(self, ccw=True):
        '''Return the counterclockwise perpendicular vector.

        If ccw is False, do the rotation in the clockwise direction.
        '''

        if ccw:
            return self._from_coords_unsafe(-self._y, self._x)
        else:
            return self._from_coords_unsafe(self._y, -self._x)

    def rotated(self, theta):
        '''Rotate vector by an angle theta around origin'''

        x, y = self
        cos_t, sin_t = self._cos(theta), self._sin(theta)
        return self._from_coords_unsafe(
            x * cos_t - y * sin_t,
            x * sin_t + y * cos_t)

    def rotated_axis(self, axis, theta):
        '''Rotate vector around given axis by the angle theta'''

        dx, dy = self - axis
        cos_t, sin_t = self._cos(theta), self._sin(theta)
        return self._from_coords_unsafe(
            dx * cos_t - dy * sin_t + axis[0],
            dx * sin_t + dy * cos_t + axis[1])

    def cross(self, other):
        '''The z component of the cross product between two bidimensional
        smallvectors'''

        a, b = other
        return self._x * b - self._y * a

    def is_null(self):
        '''Checks if vector has only null components'''

        if self._x == 0.0 and self._y == 0.0:
            return True
        else:
            return False

    def is_unity(self, tol=1e-6):
        '''Return True if the norm equals one within the given tolerance'''

        return abs(self._x * self._x + self._y * self._y - 1) < 2 * tol

    def norm(self, which=None):
        '''Returns the norm of a vector'''

        if which is None:
            return self._sqrt(self._x ** 2 + self._y ** 2)
        else:
            return Vec.norm(self, which)

    def norm_sqr(self, which=None):
        '''Returns the squared norm of a vector'''

        if which is None:
            return self._x ** 2 + self._y ** 2
        else:
            return Vec.norm(self, which)

    def __add_similar__(self, other):
        return self._from_coords_unsafe(self._x + other._x, self._y + other._y)

    def __sub_similar__(self, other):
        return self._from_coords_unsafe(self._x - other._x, self._y - other._y)

    def __mul__(self, other):
        x = self._x * other
        y = self._y * other
        if isinstance(x, self.dtype):
            return self._from_coords_unsafe(x, y)
        elif isinstance(other, self._number):
            return self.__root__(x, y)
        else:
            return NotImplemented

    def __truediv__(self, other):
        return self * (1.0 / other)


class Direction2(VecBase2D):

    '''A 2-dimensional direction/unity vector'''

    __slots__ = ()

    def __init__(self, x, y):
        norm = self._sqrt(x * x + y * y)
        if norm == 0:
            raise ValueError('null vector does not define a valid direction')

        self._x = x / norm
        self._y = y / norm

    def rotated(self, theta):
        '''Rotate vector by an angle theta around origin'''

        x, y = self
        cos_t, sin_t = self._cos(theta), self._sin(theta)
        new = Base2D.__new__(Direction2, x, y)
        new.x = x * cos_t - y * sin_t
        new.y = x * sin_t + y * cos_t
        return new


class Point2(Base2D, Point):

    '''A geometric point in 2D space'''

    __slots__ = ()


Base2D.x = Base2D._x
Base2D.y = Base2D._y

from generic.operator import add, sub


if __name__ != '__main__':
    fVec2 = Vec[2, float]

    @add.overload([Vec[2, float], Vec[2, int]])
    @add.overload([Vec[2, int], Vec[2, float]])
    def add(u, v):
        return fVec2(u.x + v.x, u.y + v._y)

    @add.overload([Vec, Vec])
    def add(u, v):
        return Vec(u.x + v.x, u.y + v.y)

    @sub.overload([Vec[2, float], Vec[2, int]])
    @sub.overload([Vec[2, int], Vec[2, float]])
    def sub(u, v):
        return fVec2(u.x - v.x, u.y - v._y)

    @sub.overload([Vec, Vec])
    def sub(u, v):
        return Vec(u.x - v.x, u.y - v.y)

    @add.overload([Vec[2, int], tuple])
    @add.overload([Vec[2, float], tuple])
    @add.overload([tuple, Vec[2, float]])
    @add.overload([tuple, Vec[2, int]])
    def add_tuple(u, v):
        x, y = u
        a, b = v
        return Vec(x + a, y + b)


if __name__ == '__main__':
    #import doctest
    # doctest.testmod()

    import time
    u = Vec(1.0, 2.0)
    v = Vec(3.0, 4.0)
    print(u)
    print(v)
    print(u == v)
    print(type(u), type(v), type(u + v))

    print('add')
    u_list = [u] * 100000
    t0 = time.time()
    S = sum(u_list, v)
    print(time.time() - t0)

    print('mul')
    t0 = time.time()
    for i in range(100000):
        v * 2
    print(time.time() - t0)

    print('div')
    t0 = time.time()
    for i in range(100000):
        v / 2
    print(time.time() - t0)

    print('add convert')
    u_list = [Vec(1, 2)] * 100000
    t0 = time.time()
    S = sum(u_list, v)
    print(time.time() - t0)

    print('add tuple')
    u_list = [(1, 2)] * 100000
    t0 = time.time()
    S = sum(u_list, v)
    print(time.time() - t0)
