"""
Abstract arithmetic operations
"""

from numbers import Number
from generic import promote_type
from generic.op import add, sub, mul, truediv, floordiv, Object
from .base import ABC, SmallVectorsBase


class AddElementWise(ABC, Object):
    """
    Implements elementwise addition and subtraction.
    """

    def __addsame__(self, other):
        return _from_data(self, [x + y for (x, y) in zip(self.flat, other.flat)])
    
    def __subsame__(self, other):
        return _from_data(self, [x - y for (x, y) in zip(self.flat, other.flat)])


class MulElementWise(ABC, Object):
    """
    Implements elementwise multiplication and division
    """

    def __mulsame__(self, other):
        return _from_data(self, [x * y for (x, y) in zip(self.flat, other.flat)])

    def __truedivsame__(self, other):
        return _from_data(self, [x / y for (x, y) in zip(self.flat, other.flat)])


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


class AddScalar(ABC, Object):
    """
    Implements scalar addition and subtraction
    """


@mul.register(MulScalar, Number)
def mul_scalar(u, number):
    return _from_data(u, [x * number for x in u])


@mul.register(Number, MulScalar)
def rmul_scalar(number, u):
    return _from_data(u, [number * x for x in u])


@truediv.register(MulScalar, Number)
def truediv_scalar(u, number):
    return _from_data(u, [x / number for x in u])


@floordiv.register(MulScalar, Number)
def floordiv_scalar(u, number):
    return _from_data(u, [x // number for x in u])

    
@add.register(AddScalar, Number)
def add_scalar(u, number):
    return _from_data(u, [x + number for x in u])


@add.register(Number, AddScalar)
def radd_scalar(number, u):
    return _from_data(u, [number + x for x in u])


@sub.register(AddScalar, Number)
def sub_scalar(u, number):
    return _from_data(u, [x - number for x in u])


@sub.register(Number, AddScalar)
def rsub_scalar(number, u):
    return _from_data(u, [number - x for x in u])


# Utility functions
def _check_scalar(obj, other, op):
    # Fasttrack most common scalar types
    if isinstance(other, (obj.dtype, float, int, Number)):
        pass
    
    elif obj.__origin__ is getattr(other, '__origin__', None):
        tname = obj.__origin__.__name__
        raise TypeError('%s instances only accept scalar multiplication' % tname)
    
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


def _from_data(obj, data):
    if isinstance(data[0], obj.dtype):
        return type(obj).fromflat(data, copy=False)
    else:
        dtype = type(data[0])
        cls = obj.__origin__[obj.shape + (dtype,)]
        return cls.fromflat(data, copy=False)
