'''
Some utility functions.
'''

from generic import promote


def get_common_base(T1, *args):
    '''Returns the most specialized common base type between T1 and T2'''

    common = object

    if not args:
        return T1
    elif len(args) == 1:
        T2 = args[0]

        for tt in T1.mro():
            if issubclass(T2, tt):
                common = tt
                break

        for tt in T2.mro():
            if issubclass(T1, tt):
                if issubclass(tt, common):
                    common = tt
                break
    else:
        T2, *args = args
        T3 = get_common_base(T1, T2)
        return get_common_base(T3, *args)

    return common


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
                tt = get_common_base(tt, type(v))
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

if __name__ == '__main__':
    assert dtype([1, 2, 3]) is int
    assert dtype([1, 2.0, 3]) is float
    assert dtype([1, 2.0, 'three']) is object
    assert shape(([1, 2, 3], [3, 4, 3])) == (2, 3)
