'''
Abstract arithmetic operations
'''

from numbers import Number
from generic import promote_type
from generic.operator import add, sub, mul, truediv
from .smallvectorsbase import ABC, SmallVectorsBase


class AddElementWise(ABC):
    '''Implements elementwise addition and subtraction'''

    def __add__(self, other):
        return (
            _check_vector(self, other, add) or 
            _from_data(self, [x + y for (x, y) in zip(self.flat, other.flat)])
        )

    def __sub__(self, other):
        return (
            _check_vector(self, other, sub) or 
            _from_data(self, [x - y for (x, y) in zip(self.flat, other.flat)])
        )
    

class MulElementWise(ABC):
    '''Implements elementwise multiplication and division'''
    

class MulScalar(ABC):
    '''Implements scalar multiplication and division'''

    def __mul__(self, other):
        return (
            _check_scalar(self, other, mul) or 
            _from_data(self, [x * other for x in self.flat])
        )
        
    def __truediv__(self, other):
        return (
            _check_scalar(self, other, truediv) or 
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


@add.overload([SmallVectorsBase, SmallVectorsBase])
def add_base(x, y):
    if x.__origin__ is y.__origin__:
        dtype = promote_type(x.dtype, y.dtype)
        return x.convert(dtype) + y.convert(dtype)
    else:
        fmt = type(x).__name__, type(y).__name__
        raise TypeError('invalid operation: add(%s, %s)' % fmt)