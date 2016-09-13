from collections import Sequence
from generic import convert
from generic.op import add, sub, mul, div
from smallvectors import (SmallVectorsBase, dtype,
                          AddElementWise, MulElementWise, AddScalar, MulScalar)
from smallvectors.core.flat import Flat

__all__ = ['Array']


class Array(AddElementWise, MulElementWise, AddScalar, MulScalar,
            SmallVectorsBase, Sequence):
    """
    Unidimensional array of uniform objects.
    """

    __parameters__ = [int, type]
    __slots__ = ('flat',)

    @classmethod
    def __abstract_new__(cls, data):
        N = len(data)
        cls = cls[N, dtype(data)]
        return cls.from_flat(data)

    @classmethod
    def from_flat(cls, data, dtype=None, copy=False, shape=None):
        if dtype is None:
            return cls(data)
        return super().from_flat(data, dtype=dtype, copy=copy, shape=shape)

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
