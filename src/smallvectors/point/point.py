import abc

from ..vector import Vec2, Vec3, Vec4
from ..vector.linear import Linear


class Point(Linear, metaclass=abc.ABCMeta):
    """
    Base class for all immutable Point types.
    """

    __slots__ = ()

    def __add__(self, other):
        if isinstance(other, Point):
            raise TypeError('cannot add two point instances')
        return super().__add__(other)

    def __sub__(self, other):
        sub = super().__sub__(other)
        if sub is NotImplemented:
            return NotImplemented
        return self._vec_class(*sub)


class Point2(Point):
    """
    A 2D point type.
    """

    __slots__ = ('_x', '_y')
    size = 2
    shape = (2,)
    _vector_class = Vec2
    __init__ = Vec2.__init__
    __getitem__ = Vec2.__getitem__
    __eq__ = Vec2.__eq__

    def __add__(self, other):
        if isinstance(other, Point):
            raise TypeError('cannot add two point instances')
        return Vec2.__add__(self, other)

    __radd__ = __add__
    __sub__ = Vec2.__sub__
    __rsub__ = Vec2.__rsub__

    def copy(self, *args, **kwargs):
        if not args and not kwargs:
            return self
        return Point2(*Vec2.copy(self, *args, **kwargs))


class Point3(Point):
    """
    A 3D point type.
    """

    __slots__ = ('_x', '_y', '_z')
    size = 3
    shape = (3,)
    _vector_class = Vec3
    __init__ = Vec3.__init__
    __getitem__ = Vec3.__getitem__
    __eq__ = Vec3.__eq__

    def __add__(self, other):
        if isinstance(other, Point):
            raise TypeError('cannot add two point instances')
        return Vec3.__add__(self, other)

    __radd__ = __add__
    __sub__ = Vec3.__sub__
    __rsub__ = Vec3.__rsub__

    def copy(self, *args, **kwargs):
        if not args and not kwargs:
            return self
        return Point3(*Vec3.copy(self, *args, **kwargs))


class Point4(Point):
    """
    A 4D point type.
    """

    __slots__ = ('_x', '_y', '_z', '_w')
    size = 4
    shape = (4,)
    _vector_class = Vec4
    __init__ = Vec4.__init__
    __getitem__ = Vec4.__getitem__
    __eq__ = Vec4.__eq__

    def __add__(self, other):
        if isinstance(other, Point):
            raise TypeError('cannot add two point instances')
        return Vec3.__add__(self, other)

    __radd__ = __add__
    __sub__ = Vec4.__sub__
    __rsub__ = Vec4.__rsub__

    def copy(self, *args, **kwargs):
        if not args and not kwargs:
            return self
        return Point4(*Vec4.copy(self, *args, **kwargs))
