# -*- coding: utf8 -*-
from __future__ import division
from collections import Sequence
from generic import convert
from generic.op import add, sub, mul, div
from smallvectors import (SmallVectorsBase, Flat, dtype,
                          AddElementWise, MulElementWise, AddScalar, MulScalar)


__all__ = ['Array']


class Array(SmallVectorsBase, Sequence, AddElementWise, MulElementWise,
            AddScalar, MulScalar):

    """Unidimensional array of uniform objects."""

    __parameters__ = [int, type]

    @classmethod
    def __abstract_new__(cls, data):
        N = len(data)
        cls = cls[N, dtype(data)]
        return cls.fromflat(data)

    def __init__(self, data):
        dtype = self.dtype
        self.flat = Flat([convert(x, dtype) for x in data], copy=False)

    def __repr__(self):
        data = ', '.join(repr(x) for x in self)
        return '%s([%s])' % (type(self).__name__, data)

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.flat)

    @classmethod
    def fromflat(cls, data, dtype=None, copy=False):
        if dtype is None:
            return cls(data)
        return super(cls.__class__, cls)(data, dtype=dtype, copy=copy)


if __name__ == '__main__':
    A = Array([1, 2, 3, 4])
    import pprint
    pprint.pprint(list(add))
    print(A)
    print(A + A)
    print(A * A)
    print(A / A)
    print(A * 2)
    print(A + 1)
    print(1 + A)
