import sys
import math
import collections
from numbers import Number
from generic import convert, promote_type
from generic.parametric import ParametricMeta, ABC, Immutable, Mutable, Any
from generic.op import Object
from generic.util import tname
from ..tools import lazy, dtype as _dtype, shape as _shape


__all__ = [
    'SmallVectorsBase', 'SmallVectorsMeta',
    'Immutable', 'Mutable',
    'Sequentiable', 'Serializable',
    'Flat', 'mFlat', 'FlatView',
]


#
# Utility functions
#
def _default_smallvectors_type_ns(params):
    '''Compute a dict with shape, ndim, dtype and size attributes computed from 
    the given parameters'''
    
    ns = {}
    if params is None:
        return ns
    
    shape = list(params[:-1])
    for i, x in enumerate(shape):
        if x is int:
            shape[i] = None
    ns['shape'] = shape = tuple(shape)
    ns['ndim'] = len(shape)

    dtype = params[-1]
    if dtype is type:
        dtype = None
    ns['dtype'] = dtype
    
    size = 1
    for dim in shape:
        if dim is None:
            ns['size'] = None
            break
        size *= dim
    else:
        ns['size'] = size
    return ns


#
# Flat and FlatView
#
class Flat(collections.Sequence):

    '''A immutable list-like object that holds a flattened data of a 
    smallvectors object.'''

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

    '''A mutable Flat object.'''

    __slots__ = ()

    def __setitem__(self, idx, value):
        self._data[idx] = value
        

class FlatView(collections.Sequence):

    '''A flat facade to some arbitrary sequence-like object.

    It accepts two functions: flat_index and owner_index that maps indexes from
    owner to flat and from flat to owner respectivelly. An explicit size can
    also be given.'''

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
                            'to support item assigment' )
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


#
# Base classes
#
class Mathematical(ABC):
    '''Defines a set of overridable mathematical functions as private methods.
    
    Class implementors should use these methods instead of functions in the 
    math module so subclasses can reuse implementations with their own math
    functions (maybe using numpy, simpy, etc).'''
     
    # Mathematical functions
    _sqrt = math.sqrt
    _sin = math.sin
    _cos = math.cos
    _tan = math.tan
    _acos = math.acos
    _asin = math.asin
    _atan = math.atan
    _atan2 = math.atan2
    _float = float
    _floating = float
    _number = (float, int, Number)


class Serializable(ABC):
    '''Base class for all objects that have a .flat attribute'''

    _nullvalue = 0
    
    def __neg__(self):
        return self.fromflat([-x for x in self], copy=False)
    
    @classmethod
    def fromflat(cls, data, copy=True, dtype=None):
        '''Initializes object from flattened data.
        
        If copy=False, it tries to recycle the flattened data whenever 
        possible. The caller is responsible for not sharing this data 
        with other mutable objects.
        
        Note
        ----
        
        For subclass implementers: this function only normalizes an iterable 
        data for final consumption.
        '''
        
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
            dtype=_dtype(data)
            
        T = cls.__origin__[cls.shape + (dtype,)]
        return T.fromflat(data, copy=copy)
    
    @classmethod
    def null(cls, shape=None):
        '''Return an object in which all components are zero'''

        null = convert(cls._nullvalue, cls.dtype)
        return cls.fromflat([null] * cls.size, shape=shape)

    def is_null(self):
        '''Checks if object has only null components'''

        null = self._nullvalue
        return all(x == null for x in self.flat)

    @lazy
    def flat(self):
        return FlatView(self)
    
    @property
    def _flatclass(self):
        return Flat if isinstance(self, Immutable) else mFlat
    
    #@abc.abstractclassmethod
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
    
    
class Sequentiable(ABC):
    '''Base class for all objects that can be iterated'''
    
    def __init__(self, *args):
        '''Directly called upon instantiation of concrete subtypes. (e.g.: at
        Vec[2, float](1, 2) rather than Vec(1, 2)).'''

        dtype = self.dtype
        if dtype is None:
            self.flat = self._flatclass(args)
        else:
            self.flat = self._flatclass([convert(x, dtype) for x in args], False)
            
    def __str__(self):
        name = type(self).__name__
        data = ', '.join(map(repr, self))
        return '%s(%s)' % (name, data)

    def __repr__(self):
        name = type(self).__origin__.__name__
        data = ', '.join(map(repr, self))
        return '%s(%s)' % (name, data)
    
    def __getitem__(self, key):
        N = len(self)
        
        if isinstance(key, int):
            if key > N:
                raise IndexError(key)
            elif key >= 0:
                for i, x in zip(self, range(N)):
                    if i == key:
                        return x
                raise IndexError(key)
            elif key < 0:
                return self[N + key]
        
        elif isinstance(key, slice):
            return [self[i] for i in range(*slice)]
        
        else:
            raise TypeError('invalid index: %r' % key)

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
    def fromdata(cls, data):
        '''Initializes from a sequence of values'''

        return cls(*data)

    
class SmallVectorsMeta(ParametricMeta):
    '''Metaclass for smallvector types'''


class SmallVectorsBase(Mathematical, Serializable, Sequentiable, Object, 
                       metaclass=SmallVectorsMeta):
    '''
    Base class for all smallvectors types.
    
    It computes the shape, size, ndim and dtype attributes from the type 
    __parameters__. It assumes that __parameters__ are always a sequence of 
    integers following by a trailing type. These integers represent the shape
    and the trailing type is the type for the scalar values.
    '''
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
        '''Create shape, size, dim, dtype'''
        
        return _default_smallvectors_type_ns(params)
        
    @staticmethod
    def __finalizetype__(cls):
        '''Assure that the resulting type has the correct shape, size, dim, 
        dtype'''
        
        # Shape parameters
        if cls.__parameters__ is None or cls.shape is None:
            default_ns = _default_smallvectors_type_ns(cls.__parameters__)
            for k, v in default_ns.items():
                if getattr(cls, k, None) is None:
                    setattr(cls, k, v)
                    
        # Pick up flat object
        flat = mFlat if issubclass(cls, Mutable) else Flat
        cls.__flat__ = flat
        
        # Floating parameter
        if cls.dtype is not None:
            cls._floating = promote_type(cls._float, cls.dtype)
        
        assert cls.dtype is not Any, cls

    @classmethod
    def __abstract_new__(cls, *args, shape=None, dtype=None):  # @NoSelf
        '''
        This function is called when user tries to instatiate an abstract
        type. It just finds the proper concrete type and instantiate it.
        '''
        if dtype is None:
            dtype = _dtype(args)
        if shape is None:
            shape = _shape(args)
        return cls[shape + (dtype,)](*args)

    def __flatlen__(self):
        return self.size

    def convert(self, dtype):
        '''Convert object to the given data type'''

        cls = type(self)
        if dtype is self.dtype:
            return self
        else:
            return cls.__origin__(*self, dtype=dtype)


#
# Override Mutable and Immutable base classes
#
def get_sibling_classes(cls):
    '''Helper function for finding the Mutable/Immutable pair of classes.'''
    
    parent = cls.mro()[1]
    mod = sys.modules[cls.__module__]
    names = (x for x in dir(mod) if not x.startswith('_'))
    objects = (getattr(mod, x) for x in names)
    types = (x for x in objects if isinstance(x, type))
    siblings = [T for T in types if issubclass(T, parent)]
    return siblings
    
    
class Mutable(Mutable):
    '''Base class for all mutable types'''
    
    def mutable(self):
        '''Return a mutable copy of object'''
        
        return self.copy()
    
    def immutable(self):
        '''Return an immutable copy of object'''
        
        try:
            cls = self._immutable_
        except AttributeError:
            siblings = get_sibling_classes(type(self))
            immutable = [T for T in siblings if issubclass(T, Immutable)]
            assert len(immutable) == 1
            cls = type(self)._immutable_ = immutable[0]
        return cls(*self)
    
    
class Immutable(Immutable):
    '''Base class for all immutable types'''
    
    def mutable(self):
        '''Return a mutable copy of object'''
        
        try:
            cls = self._mutable_
        except AttributeError:
            siblings = get_sibling_classes(type(self))
            mutable = [T for T in siblings if issubclass(T, Mutable)]
            assert len(mutable) == 1
            cls = type(self)._mutable_ = mutable[0]
        return cls(*self)
    
    def immutable(self):
        '''Return an immutable copy of object'''
        
        return self
