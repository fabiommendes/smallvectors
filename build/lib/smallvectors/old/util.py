#-*- coding: utf8 -*-

from smallvectors import Vec1, Vec2, Vec3, Vec4, VecND, Vec
from smallvectors import Direction2, Direction3, Direction4, DirectionND
from smallvectors import Point2, Point3, Point4, PointND, Point

###############################################################################
#                         Propriedades vetoriais
###############################################################################


class VecSlot(object):

    '''A slot-like property that holds a vector object'''

    __slots__ = ['getter', 'setter']

    def __init__(self, slot):
        self.setter = slot.__set__
        self.getter = slot.__get__

    def __get__(self, obj, tt):
        return self.getter(obj, tt)

    def __set__(self, obj, value):
        if not isinstance(value, Vec2):
            value = Vec2.from_seq(value)
        self.setter(obj, value)

    def update_class(self, tt=None, *args):
        '''Update all enumerated slots/descriptors in class to be VecSlots'''

        if tt is None:

            def decorator(tt):
                self.update_class(tt, *args)
                return tt
            return decorator

        for sname in args:
            slot = getattr(tt, sname)
            setattr(tt, sname, VecSlot(slot))


#
# Utility functions
#
VECTOR_TYPES = [Vec1, Vec2, Vec3, Vec4]


def asvector(u, accept_points=False):
    '''Convert object to vector.

    If accept_points is False (default), it raises a TypeError when trying to
    convert point objects. Generally we want to keep points and vectors
    separate, however sometimes it is useful to be able to explicitly convert
    between them.'''

    if isinstance(u, (Vec)):
        return u
    if isinstance(u, (tuple, list)):
        return VecND.from_seq(u)
    if isinstance(u, Point) and not accept_points:
        raise TypeError('cannot convert point to vector')
    return VecND.from_seq(u)


def vector(*args):
    '''Create vector from components'''

    N = len(args)
    if 1 <= N <= 4:
        return VECTOR_TYPES[N - 1](*args)
    elif N == 0:
        raise TypeError('cannot create a vector with zero components')
    else:
        return VecND(*args)


def norm(vec):
    '''Return the norm of a vector'''

    try:
        return vec.norm()
    except AttributeError:
        if isinstance(vec, tuple):
            return m.sqrt(sum(x * x for x in vec))
        else:
            tname = type(vec).__name__
            raise TypeError('norm is not defined for %s object' % tname)


def normalize(obj):
    '''Return a normalized version of vector or tuple'''

    try:
        return obj.normalize()
    except AttributeError:
        if isinstance(obj, tuple):
            return asvector(obj)
        else:
            tname = type(obj).__name__
            raise TypeError('normalize is not defined for %s object' % tname)


###############################################################################
# Late binding
###############################################################################
VecND._dim2 = Vec2
VecND._dim3 = Vec3
VecND._dim4 = Vec4
DirectionND._dim2 = Direction2
DirectionND._dim3 = Direction3
DirectionND._dim4 = Direction4
PointND._dim2 = Point2
PointND._dim3 = Point3
PointND._dim4 = Point4

import smallvectors.old
import smallvectors

smallvectors.old.asvector = asvector
smallvectors.old.vector = vector
smallvectors.asvector = asvector
smallvectors.vector = vector
