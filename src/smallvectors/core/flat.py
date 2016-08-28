import collections
from typing import Any

from generic import convert
from generic.parametric import ABC, Immutable
from generic.util import tname

from smallvectors.tools import dtype as _dtype, lazy


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

    __slots__ = ('owner',)

    def __init__(self, owner):
        self.owner = owner

    def __repr__(self):
        return 'flat([%s])' % (', '.join(map(repr, self)))

    def __iter__(self):
        for x in self.owner.__flatiter__():
            yield x

    def __len__(self):
        return self.owner.__flatlen__()

    def __getitem__(self, key):
        owner = self.owner
        try:
            if isinstance(key, int):
                return owner.__flatgetitem__(key)
            else:
                getter = owner.__flatgetitem__
                indices = range(*key.indices(self.owner.size))
                return [getter(i) for i in indices]
        except AttributeError:
            N = owner.size

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
        if isinstance(self.owner, Immutable):
            raise TypeError('cannot change immutable object')
        try:
            setter = self.owner.__flatsetitem__
        except AttributeError:
            raise TypeError('object must implement __flatsetitem__ in order '
                            'to support item assigment')
        else:
            N = self.owner.size

            if isinstance(key, int):
                if key < 0:
                    key = N - key
                for i, _ in enumerate(self):
                    if i == key:
                        return setter(i, value)
                else:
                    raise IndexError(key)

            elif isinstance(key, slice):
                indices = range(*key.indices(N))
                return [setter(i, x) for i, x in zip(indices, value)]

            else:
                raise IndexError('invalid index: %r' % key)


class Flatable(ABC):
    """
    Base class for all objects that have a .flat attribute
    """

    _nullvalue = 0

    def __neg__(self):
        return self.fromflat([-x for x in self], copy=False)

    @classmethod
    def fromflat(cls, data, copy=True, dtype=None):
        """
        Initializes object from flattened data.

        If copy=False, it tries to recycle the flattened data whenever
        possible. The caller is responsible for not sharing this data
        with other mutable objects.

        Note
        ----

        For subclass implementers: this function only normalizes an iterable
        data for final consumption.
        """

        if cls.__concrete__:
            if dtype is None or dtype is cls.dtype:
                new = object.__new__(cls)
                new.flat = cls.__flat__(data, copy)
                return new
        elif cls.size is None:
            raise TypeError('shapeless types cannot instanciate objects')

        dtype = dtype or cls.dtype
        if dtype is Any or dtype is None:
            data = list(data)
            dtype = _dtype(data)

        T = cls.__origin__[cls.shape + (dtype,)]
        return T.fromflat(data, copy=copy)

    @classmethod
    def null(cls, shape=None):
        """Return an object in which all components are zero"""

        null = convert(cls._nullvalue, cls.dtype)
        return cls.fromflat([null] * cls.size, shape=shape)

    def is_null(self):
        """Checks if object has only null components"""

        null = self._nullvalue
        return all(x == null for x in self.flat)

    @lazy
    def flat(self):
        return FlatView(self)

    @property
    def _flatclass(self):
        return Flat if isinstance(self, Immutable) else mFlat

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