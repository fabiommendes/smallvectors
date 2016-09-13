from numbers import Number

from generic import promote_type, convert
from generic.op import add, sub, mul, truediv, floordiv, Object

from smallvectors.core import ABC
from smallvectors.core.base import SmallVectorsBase


class AddElementWise(ABC, Object):
    """
    Implements elementwise addition and subtraction.
    """

    __slots__ = ()

    def __addsame__(self, other):
        return fromflat(self,
                        [x + y for (x, y) in zip(self.flat, other.flat)])

    def __subsame__(self, other):
        return fromflat(self,
                        [x - y for (x, y) in zip(self.flat, other.flat)])


class mAddElementWise(AddElementWise):
    """
    Inplace addition operators
    """

    def __iadd__(self, other):
        other = convert(other, type(self))
        for i, x in other.flat:
            self.flat[i] += x
        return self

    def __isub__(self, other):
        other = convert(other, type(self))
        for i, x in other.flat:
            self.flat[i] -= x
        return self


class MulElementWise(ABC, Object):
    """
    Implements elementwise multiplication and division
    """

    __slots__ = ()

    def __mulsame__(self, other):
        return fromflat(self,
                        [x * y for (x, y) in zip(self.flat, other.flat)])

    def __truedivsame__(self, other):
        return fromflat(self,
                        [x / y for (x, y) in zip(self.flat, other.flat)])


class mMulElementWise(MulElementWise):
    """
    Inplace multiplication operators.
    """

    def __imul__(self, other):
        other = convert(other, type(self))
        for i, x in other.flat:
            self.flat[i] *= x
        return self

    def __itruediv__(self, other):
        other = convert(other, type(self))
        for i, x in other.flat:
            self.flat[i] /= x
        return self

    def __ifloordiv__(self, other):
        other = convert(other, type(self))
        for i, x in other.flat:
            self.flat[i] /= x
        return self


@add.register(AddElementWise, AddElementWise, factory=True)
def add_elementwise_factory(argtypes, restype):
    T, S = argtypes
    if issubclass(T, S):
        return S.__addsame__
    if issubclass(S, T):
        return T.__addsame__
    if T.__origin__ is not S.__origin__:
        return NotImplemented
    if T.shape != S.shape:
        return NotImplemented

    def func(u, v):
        return u.convert(dtype) + v.convert(dtype)

    dtype = promote_type(T.dtype, S.dtype)
    return func


@sub.register(AddElementWise, AddElementWise, factory=True)
def sub_elementwise_factory(argtypes, restype):
    T, S = argtypes
    if issubclass(T, S):
        return S.__subsame__
    if issubclass(S, T):
        return T.__subsame__
    if T.__origin__ is not S.__origin__:
        return NotImplemented
    elif T.shape != S.shape:
        return NotImplemented

    def func(u, v):
        return u.convert(dtype) - v.convert(dtype)

    dtype = promote_type(T.dtype, S.dtype)
    return func


@mul.register(MulElementWise, MulElementWise, factory=True)
def mul_elementwise_factory(argtypes, restype):
    T, S = argtypes
    if issubclass(T, S):
        return S.__mulsame__
    if issubclass(S, T):
        return T.__mulsame__
    if T.__origin__ is not S.__origin__:
        return NotImplemented
    if T.shape != S.shape:
        return NotImplemented

    def func(u, v):
        return u.convert(dtype) * v.convert(dtype)

    dtype = promote_type(T.dtype, S.dtype)
    return func


@truediv.register(MulElementWise, MulElementWise, factory=True)
def truediv_elementwise_factory(argtypes, restype):
    T, S = argtypes
    if issubclass(T, S):
        return S.__truedivsame__
    if issubclass(S, T):
        return T.__truedivsame__
    if T.__origin__ is not S.__origin__:
        return NotImplemented
    elif T.shape != S.shape:
        return NotImplemented

    def func(u, v):
        return u.convert(dtype) / v.convert(dtype)

    dtype = promote_type(T.dtype, S.dtype)
    return func


class MulScalar(ABC, Object):
    """
    Implements scalar multiplication and division.
    """

    __slots__ = ()


class mMulScalar(MulScalar):
    """
    Inplace multiplication.
    """

    __slots__ = ()

    def __imul__(self, other):
        other = convert(other, self.dtype)
        for i, x in enumerate(self.flat):
            self.flat[i] *= x

    def __itruediv__(self, other):
        other = convert(other, self.dtype)
        for i, x in enumerate(self.flat):
            self.flat[i] /= x

    def __ifloordiv__(self, other):
        other = convert(other, self.dtype)
        for i, x in enumerate(self.flat):
            self.flat[i] //= x


class AddScalar(ABC, Object):
    """
    Implements scalar addition and subtraction
    """

    __slots__ = ()


class mAddScalar(AddScalar):
    """
    Inplace addition.
    """

    __slots__ = ()

    def __iadd__(self, other):
        other = convert(other, self.dtype)
        for i, x in enumerate(self.flat):
            self.flat[i] += x

    def __isub__(self, other):
        other = convert(other, self.dtype)
        for i, x in enumerate(self.flat):
            self.flat[i] -= x


@mul.register(MulScalar, Number)
def mul_scalar(u, number):
    return fromflat(u, [x * number for x in u])


@mul.register(Number, MulScalar)
def rmul_scalar(number, u):
    return fromflat(u, [number * x for x in u])


@truediv.register(MulScalar, Number)
def truediv_scalar(u, number):
    return fromflat(u, [x / number for x in u])


@floordiv.register(MulScalar, Number)
def floordiv_scalar(u, number):
    return fromflat(u, [x // number for x in u])


@add.register(AddScalar, Number)
def add_scalar(u, number):
    return fromflat(u, [x + number for x in u])


@add.register(Number, AddScalar)
def radd_scalar(number, u):
    return fromflat(u, [number + x for x in u])


@sub.register(AddScalar, Number)
def sub_scalar(u, number):
    return fromflat(u, [x - number for x in u])


@sub.register(Number, AddScalar)
def rsub_scalar(number, u):
    return fromflat(u, [number - x for x in u])


# Utility functions
def _check_scalar(obj, other, op):
    # Fast-track most common scalar types
    if isinstance(other, (obj.dtype, float, int, Number)):
        pass

    elif obj.__origin__ is getattr(other, '__origin__', None):
        tname = obj.__origin__.__name__
        raise TypeError(
            '%s instances only accept scalar multiplication' % tname)

    elif isinstance(other, (list, tuple, SmallVectorsBase)):
        return op(obj, other)


def _check_vector(obj, other, op):
    try:
        if obj.__class__ is other.__class__:
            return
        if obj.__origin__ is other.__origin__:
            if obj.shape != other.shape:
                data = obj.shape, other.shape
                raise ValueError('incompatible shapes: %s and %s' % data)
            return
    except AttributeError:
        return op(obj, other)


def fromflat(obj, data):
    if isinstance(data[0], obj.dtype):
        return type(obj).from_flat(data, copy=False)
    else:
        dtype = type(data[0])
        cls = obj.__origin__[obj.shape + (dtype,)]
        return cls.from_flat(data, copy=False)
