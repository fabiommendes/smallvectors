# -*- coding: utf8 -*-
import itertools
from generic import promote


def flatten(L, ndim=None):
    '''Flatten the given list of lists and return a tuple of (data, *dimensions)'''
    
    flat = list(L)
    indices = [len(flat)]
    while ndim != 1:
        try:
            flat = [list(x) for x in flat]
        except TypeError:
            if ndim is None:
                return (flat,) + tuple(indices)
            raise ValueError('expected more levels in data')
        
        sizes = set(map(len, flat))
        if len(sizes) > 1:
            raise ValueError('inconsistent shapes')
        indices.append(sizes.pop())
        flat = list(itertools.chain(*flat))
        if ndim is not None:
            ndim -= 1
        
    return (flat,) + tuple(indices)
    

def commonbase(T1, *args):
    '''Returns the most specialized common base type between T1 and T2'''

    common = object

    if not args:
        return T1
    else:
        T2 = args[0]
        assert isinstance(T2, type), ('not a type: %s' % repr(T2))

        for tt in T1.mro():
            if issubclass(T2, tt):
                common = tt
                break

        for tt in T2.mro():
            if issubclass(T1, tt):
                if issubclass(tt, common):
                    common = tt
                break
    
        if len(args) == 1:
            return common
        else:
            return commonbase(common, *args[1:])


def dtype(values):
    '''Return the dtype for a group of values'''

    if not values:
        raise TypeError('trying to find the type of an empty list')

    values = iter(values)
    value = next(values)
    tt = type(value)

    for v in values:
        if not isinstance(v, tt):
            try:
                value, v = promote(value, v)
                tt = type(value)
            except TypeError:
                tt = commonbase(tt, type(v))
    return tt


def shape(data):
    '''Return the shape of a list, list of lists, list of list of lists, etc.

    Non-uniform shapes raises an ValueError.
    '''
    try:
        return data.shape
    except AttributeError:
        shape = []
        while True:
            try:
                shape.append(len(data))
                data = data[0]
            except TypeError:
                break

        return tuple(shape)

class lazy(object):

    '''Lazy accessor'''

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        value = self.func(obj)
        setattr(obj, self.func.__name__, value)
        return value


def sign(x):
    '''Returns -1, 0, or 1 for the sign of a number'''

    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
