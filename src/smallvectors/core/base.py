from generic import convert, promote_type
from generic.op import Object
from generic.parametric import ParametricMeta, ABC, Immutable as _Immutable, \
    Mutable as _Mutable, Any

from smallvectors.core.flat import Flat, mFlat, Flatable
from smallvectors.core.mixins import MathFunctionsMixin
from smallvectors.core.util import default_smallvectors_type_ns, \
    get_sibling_classes
from ..tools import dtype as _dtype, shape as _shape

__all__ = [
    'SmallVectorsBase', 'SmallVectorsMeta',
    'Immutable', 'Mutable',
    'Sequentiable', ]


class Sequentiable(ABC):
    """
    Base class for all objects that support iteration.
    """

    __origin__ = None
    _flatclass = NotImplemented
    dtype = NotImplemented
    size = NotImplemented
    shape = NotImplemented
    is_immutable = False

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

    def __str__(self):
        name = type(self).__name__
        data = ', '.join(map(repr, self))
        return '%s(%s)' % (name, data)

    def __repr__(self):
        name = type(self).__origin__.__name__
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
            start, stop, step = key
            return [self[i] for i in range(start, stop, step)]

        else:
            raise TypeError('invalid index: %r' % key)

    def __setitem__(self, key, value):
        if self.is_immutable:
            raise KeyError('cannot set coordinate of immutable object')
        else:
            raise NotImplementedError('subclass must implement __setitem__')

    def __len__(self):
        return NotImplemented

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

    @classmethod
    def from_data(cls, data):
        """
        Initializes from a sequence of values.
        """

        return cls(*data)


class SmallVectorsMeta(ParametricMeta):
    """
    Metaclass for smallvector types.
    """


# noinspection PyAbstractClass
class SmallVectorsBase(MathFunctionsMixin, Flatable, Sequentiable, Object,
                       metaclass=SmallVectorsMeta):
    """
    Base class for all smallvectors types.
    
    It computes the shape, size, ndim and dtype attributes from the type 
    __parameters__. It assumes that __parameters__ are always a sequence of 
    integers following by a trailing type. These integers represent the shape
    and the trailing type is the type for the scalar values.
    """

    __abstract__ = True
    __parameters__ = None
    __slots__ = ()

    # Basic parameters of smallvectors types
    shape = None
    size = None
    ndim = None
    dtype = None

    @classmethod
    def __preparenamespace__(cls, params):
        """
        Create shape, size, dim, dtype.
        """

        return default_smallvectors_type_ns(params)

    @staticmethod
    def __finalizetype__(cls):
        """
        Assure that the resulting type has the correct shape, size, dim,
        dtype
        """

        # Shape parameters
        if cls.__parameters__ is None or cls.shape is None:
            default_ns = default_smallvectors_type_ns(cls.__parameters__)
            for k, v in default_ns.items():
                if getattr(cls, k, None) is None:
                    setattr(cls, k, v)

        # Pick up flat object
        flat = mFlat if issubclass(cls, _Mutable) else Flat
        cls.__flat__ = flat

        # Floating parameter
        if cls.dtype is not None:
            cls._floating = promote_type(cls._float, cls.dtype)

        assert cls.dtype is not Any, cls

    @classmethod
    def __abstract_new__(cls, *args, shape=None, dtype=None):
        """
        This function is called when user tries to instantiate an abstract
        type. It just finds the proper concrete type and instantiate it.
        """
        if dtype is None:
            dtype = _dtype(args)
        if shape is None:
            shape = _shape(args)
        return cls[shape + (dtype,)](*args)

    def __flatlen__(self):
        return self.size

    def convert(self, dtype):
        """
        Return a copy of object converted to the given data type.
        """

        cls = type(self)
        if dtype is self.dtype:
            return self.copy()
        else:
            return cls.__origin__(*self, dtype=dtype)

    def copy(self):
        return NotImplemented


class Mutable(_Mutable):
    """
    Base class for all mutable types.
    """

    def __getstate__(self):
        return NotImplemented

    _immutable_ = NotImplemented

    @property
    def _mutable_(self):
        return self.__class__

    @property
    def is_mutable(self):
        return True

    @property
    def is_immutable(self):
        return False

    def mutable(self):
        """
        Return a mutable copy of object.
        """

        return self.copy()

    def immutable(self):
        """
        Return an immutable copy of object.
        """

        try:
            cls = self._immutable_
        except AttributeError:
            siblings = get_sibling_classes(type(self))
            immutable = [T for T in siblings if issubclass(T, Immutable)]
            assert len(immutable) == 1
            cls = type(self)._immutable_ = immutable[0]
        return cls(*self.__getstate__())

    def copy(self):
        return NotImplemented


class Immutable(_Immutable):
    """
    Base class for all immutable types.
    """

    def __getstate__(self):
        return NotImplemented

    _mutable_ = NotImplemented

    @property
    def _immutable_(self):
        return self.__class__

    @property
    def is_mutable(self):
        return False

    @property
    def is_immutable(self):
        return True

    def mutable(self):
        """
        Return a mutable copy of object.
        """

        try:
            cls = self._mutable_
        except AttributeError:
            siblings = get_sibling_classes(type(self))
            mutable = [T for T in siblings if issubclass(T, Mutable)]
            assert len(mutable) == 1
            cls = type(self)._mutable_ = mutable[0]
        return cls(*self.__getstate__())

    def immutable(self):
        """
        Return an immutable copy of object.
        """

        return self

    def copy(self):
        return self
