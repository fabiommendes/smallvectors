from generic import convert

from smallvectors.core import ABC
from smallvectors.utils import shape as _shape


def slicerange(slice, obj):
    """
    Return a range from a slice.
    """

    start = slice.start or 0
    stop = slice.stop
    step = slice.step or 1

    if stop is None:
        return range(start, len(obj), step)
    return range(start, stop, step)


class Sequentiable(ABC):
    """
    Base class for all objects that support iteration.
    """

    __slots__ = ()
    __origin__ = None
    _flatclass = NotImplemented
    dtype = NotImplemented
    size = NotImplemented
    shape = NotImplemented

    @classmethod
    def from_data(cls, data):
        """
        Initializes from a sequence of values.
        """

        return cls(*data)

    def __init__(self, *args):
        """
        Directly called upon instantiation of concrete subtypes. (e.g.: at
        Vec[2, float](1, 2) rather than Vec(1, 2))."""

        dtype = self.dtype
        if dtype is None:
            self.flat = self._flatclass(args)
        else:
            self.flat = self._flatclass([convert(x, dtype) for x in args],
                                        False)

    def __repr__(self):
        try:
            name = self.__class__.__origin__.__name__
        except AttributeError:
            name = self.__class__.__name__
        data = ', '.join(map(repr, self))
        return '%s(%s)' % (name, data)

    def __getitem__(self, key):
        size = len(self)

        if isinstance(key, int):
            if key > size or key < -size:
                raise IndexError(key)
            elif key >= 0:
                return self.__getitem_simple__(key)
            elif key < 0:
                return self[size + key]

        elif isinstance(key, slice):
            return [self[i] for i in slicerange(key, self)]

        else:
            raise TypeError('invalid index: %r' % key)

    def __setitem__(self, key, value):
        if self.is_immutable():
            raise KeyError('cannot set coordinate of immutable object')
        else:
            tname = self.__class__.__name__
            raise NotImplementedError('%s must implement __setitem__' % tname)

    def __len__(self):
        raise NotImplementedError

    def __getstate__(self):
        return tuple(self)

    def __getitem_simple__(self, key):
        for i, x in zip(self, range(len(self))):
            if i == key:
                return x
        raise IndexError(key)

    def __eq__(self, other):
        if self.shape != _shape(other):
            return False

        if self.__origin__ is getattr(other, '__origin__', None):
            if self.shape != other.shape:
                return False
            return all(x == y for (x, y) in zip(self.flat, other.flat))
        else:
            return all(x == y for (x, y) in zip(self, other))

    def __nonzero__(self):
        return bool(self.size)

    def __bool__(self):
        return bool(self.size)
