import collections
from typing import Any

from generic import convert
from generic.parametric import Mutable, Immutable
from generic.util import tname

from smallvectors.core import ABC
from smallvectors.utils import dtype as _dtype


class Flat(collections.Sequence):
    """
    A immutable list-like object that holds a flattened data of a
    smallvectors object.
    """

    __slots__ = ('_data',)

    def __init__(self, data, copy=True):
        if copy:
            self._data = list(data)
        else:
            self._data = data

    def __repr__(self):
        return 'flat([%s])' % (', '.join(map(repr, self)))

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, idx_or_slice):
        return self._data[idx_or_slice]

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        if isinstance(other, (Flat, list, tuple)):
            if len(self) == len(other):
                return all(x == y for x, y in zip(self, other))
            return False
        return NotImplemented


class mFlat(Flat):
    """
    A mutable Flat object.
    """

    __slots__ = ()

    def __setitem__(self, idx, value):
        self._data[idx] = value


class FlatView(collections.Sequence):
    """
    A flat façade to some arbitrary sequence-like object.

    It accepts two functions: flat_index and owner_index that maps indexes from
    owner to flat and from flat to owner respectively. An explicit size can
    also be given.
    """

    __slots__ = ('_object',)

    def __init__(self, owner):
        self._object = owner

    def __repr__(self):
        return 'flat([%s])' % (', '.join(map(repr, self)))

    def __iter__(self):
        for x in self._object.__flatiter__():
            yield x

    def __len__(self):
        return self._object.__flatlen__()

    def __getitem__(self, key):
        obj = self._object
        try:
            if isinstance(key, int):
                return obj.__flatgetitem__(key)
            else:
                getter = obj.__flatgetitem__
                indices = range(*key.indices(self._object.size))
                return [getter(i) for i in indices]
        except AttributeError:
            N = obj.size

            if isinstance(key, int):
                if key < 0:
                    key = N - key
                for i, x in enumerate(self):
                    if i == key:
                        return x
                else:
                    raise IndexError(key)

            elif isinstance(key, slice):
                indices = range(*key.indices(N))
                return [self[i] for i in indices]

            else:
                raise IndexError('invalid index: %r' % key)

    def __setitem__(self, key, value):
        obj = self._object
        if isinstance(obj, Immutable):
            raise TypeError('cannot change immutable object')
        try:
            setter = obj.__flatsetitem__
        except AttributeError:
            raise TypeError('object must implement __flatsetitem__ in order '
                            'to support item assignment')
        else:
            N = obj.__flatlen__()
            assert not isinstance(N, type(NotImplemented))

            if isinstance(key, int):
                if key < -N:
                    raise IndexError(key)
                elif key < 0:
                    key = N - key
                setter(key, value)

            elif isinstance(key, slice):
                indices = range(*key.indices(N))
                return [setter(i, x) for i, x in zip(indices, value)]

            else:
                raise IndexError('invalid index: %r' % key)


class Flatable(ABC):
    """
    Base class for all objects that have a .flat attribute
    """

    __slots__ = ()
    _nullvalue = 0
    __concrete__ = NotImplemented
    __origin__ = NotImplemented
    size = NotImplemented
    dtype = NotImplemented
    shape = NotImplemented

    def __neg__(self):
        return self.from_flat([-x for x in self], copy=False)

    @classmethod
    def from_flat(cls, data):
        """
        Initializes object from flattened data.

        If copy=False, it tries to recycle the flattened data whenever
        possible. The caller is responsible for not sharing this data
        with other mutable objects.

        Note:
            This function only normalizes an iterable data for final
            consumption.
        """

        raise NotImplementedError

    @classmethod
    def null(cls, shape=None):
        """
        Return an object in which all components are zero.
        """

        null = convert(cls._nullvalue, cls.dtype)
        return cls.from_flat([null] * cls.size, shape=shape)

    def is_null(self):
        """Checks if object has only null components"""

        null = self._nullvalue
        return all(x == null for x in self.flat)

    @property
    def flat(self):
        return FlatView(self)

    @property
    def _flatclass(self):
        return Flat if isinstance(self, Immutable) else mFlat

    @classmethod
    def __flat__(cls, data, copy):
        if issubclass(cls, Mutable):
            return mFlat(data, copy)
        else:
            return Flat(data, copy)

    def __flatiter__(self):
        raise NotImplementedError(
            '%s objects does not implement a __flatiter__ method' % tname(self)
        )

    def __flatlen__(self):
        raise NotImplementedError(
            '%s objects does not implement a __flatlen__ method' % tname(self)
        )

    def __flatgetitem__(self, idx):
        raise NotImplementedError(
            '%s objects does not implement a __flatgetitem__ method;\n'
            'This function only requires positive scalar indexing to work.'
            % tname(self)
        )

    def __flatsetitem__(self, idx, value):
        raise NotImplementedError(
            '%s objects does not implement a __flatsetitem__ method;\n'
            'This function only requires positive scalar indexing to work.'
            % tname(self)
        )
