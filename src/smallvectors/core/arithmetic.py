'''
Created on 26 de out de 2015

@author: chips
'''
from numbers import Number
from generic import promote_type
from generic.operator import add, add_safe, sub_safe, mul_safe, truediv_safe
from .smallvectorsbase import ABC, SmallVectorsBase


class AddElementWise(ABC):
    '''Implements elementwise addition and subtraction'''

    def __add__(self, other):
        return (
            _check_vector(self, other, add_safe) or 
            _from_data(self, [x + y for (x, y) in zip(self.flat, other.flat)])
        )

    def __radd__(self, other):
        return (
            _check_vector(self, other, radd_safe) or 
            _from_data(self, [y + x for (x, y) in zip(self.flat, other.flat)])
        )

    def __sub__(self, other):
        return (
            _check_vector(self, other, sub_safe) or 
            _from_data(self, [x - y for (x, y) in zip(self.flat, other.flat)])
        )
    
    def __rsub__(self, other):
        return (
            _check_vector(self, other, rsub_safe) or 
            _from_data(self, [y - x for (x, y) in zip(self.flat, other.flat)])
        )


class MulElementWise(ABC):
    '''Implements elementwise multiplication and division'''
    

class MulScalar(ABC):
    '''Implements scalar multiplication and division'''

    def __mul__(self, other):
        return (
            _check_scalar(self, other, mul_safe) or 
            _from_data(self, [x * other for x in self.flat])
        )
        
    def __rmul__(self, other):
        return (
            _check_scalar(self, other, rmul_safe) or 
            _from_data(self, [other * x for x in self.flat])
        )
        
    def __truediv__(self, other):
        return (
            _check_scalar(self, other, truediv_safe) or 
            _from_data(self, [x / other for x in self.flat])
        )    
    
class AddScalar(ABC):
    '''Implements scalar addition and subtraction'''


#
# Utility functions
# 
def _check_scalar(obj, other, op):
    # Fasttrack most common scalar types
    if isinstance(other, (obj.dtype, float, int, Number)):
        return
    
    elif obj.__origin__ is getattr(other, '__origin__', None):
        tname = obj.__origin__.__name__
        raise TypeError('%s instances only accept scalar multiplication' % tname)
    
    elif isinstance(other, (list, tuple, SmallVectorsBase)):
        return op(obj, other)

def _check_vector(obj, other, op):
    try:
        if obj.__origin__ is other.__origin__:
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

#
# Arithmetic function overloads
#
def radd_safe(x, y):
    return add_safe(y, x)

def rsub_safe(x, y):
    return sub_safe(y, x)

def rmul_safe(x, y):
    return mul_safe(y, x)


@add.overload([SmallVectorsBase, SmallVectorsBase])
def add_base(x, y):
    if x.__origin__ is y.__origin__:
        dtype = promote_type(x.dtype, y.dtype)
        return x.convert(dtype) + y.convert(dtype)
    else:
        fmt = type(x).__name__, type(y).__name__
        raise TypeError('invalid operation: add(%s, %s)' % fmt)