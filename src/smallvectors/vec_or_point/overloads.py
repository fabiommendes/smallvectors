'''
Created on 15/09/2015

@author: chips
'''

import operator
from generic import set_promotion, set_conversion
from generic.operator import add, sub
from . import Vec, mVec, Point, mPoint, Direction
from . import VecOrPoint, AnyVec, AnyPoint, asvector

__all__ = []

# Conversions and promotions between vec types and tuples/lists
set_conversion(VecOrPoint, tuple, tuple)
set_conversion(VecOrPoint, list, list)
for T in [Vec, mVec, Point, mPoint, Direction]:
    set_conversion(tuple, T, T)
    set_conversion(list, T, T)


@set_promotion(VecOrPoint, tuple, symmetric=True)
@set_promotion(VecOrPoint, list, symmetric=True)
def promote(u, v):
    return u, u.__root__.from_seq(v)


#
# Addition of tuples, lists and points to vectors result in vectors
#
def asvector_overload(op, tt):
    vector = asvector
    real_op = getattr(operator, op.__name__)

    @op.overload((VecOrPoint, tt))
    def overload(u, v):
        return real_op(u, vector(v))

    @op.overload((tt, VecOrPoint))
    def overload(u, v):  # @DuplicatedSignature
        return real_op(vector(u), v)

for op in [add, sub]:
    for tt in [tuple, list, AnyPoint]:
        asvector_overload(op, tt)


#
# Addition of tuples/lists and Points results in vectors
#
@add.overload((AnyPoint, tuple))
@add.overload((AnyPoint, list))
@add.overload((AnyPoint, AnyVec))
def add(x, y):
    return x + asvector(y)


@add.overload((tuple, AnyPoint))
@add.overload((list, AnyPoint))
@add.overload((AnyVec, AnyPoint))
def add(x, y):
    return asvector(x) + y
