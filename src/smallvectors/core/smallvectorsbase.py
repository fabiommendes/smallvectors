import math
from numbers import Number
from generic import convert
from .parametric import ParametricMeta, ABC, Immutable, Mutable, Any
from .flatobject import Flat, mFlat, FlatView
from .util import lazy, dtype as _dtype, shape as _shape


__all__ = [
    'SmallVectorsBase',
    'SmallVectorsMeta',
    'Immutable',
    'Mutable',
    'Sequentiable',
    'Serializable',
]


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


class Serializable(ABC):
    '''Base class for all objects that have a .flat attribute'''

    def __neg__(self):
        return self.fromflat([-x for x in self], copy=False)
    
    def __nonzero__(self):
        return bool(self.size)
    
    @classmethod
    def fromflat(cls, data, copy=True, dtype=None):
        '''Initializes object from flattened data.
        
        If copy=False, it tries to recycle the flattened data whenever 
        possible. The caller is responsible for not sharing this data 
        with other mutable objects.'''
        
        if cls.__concrete__:
            if dtype is None or dtype is cls.dtype:
                new = object.__new__(cls)
                if issubclass(cls, Mutable):
                    new.flat = Flat(data, copy)
                else:
                    new.flat = mFlat(data, copy)
                return new
        
        dtype = dtype or cls.dtype    
        if dtype is Any or dtype is None:
            dtype=_dtype(data)
        T = cls.__origin__[cls.shape + (dtype,)]
        return T.fromflat(data, copy=copy)
    
    @classmethod
    def null(cls, shape=None):
        '''Return an object in which all components are zero'''

        if cls._null_value is not None:
            null = cls._null_value
        else:
            null = convert(0, cls.dtype)
        return cls.fromflat([null] * cls.size, shape=shape)
    
    @lazy
    def flat(self):
        return FlatView(self)
    
    def is_null(self):
        '''Checks if object has only null components'''

        return all(x == 0.0 for x in self.flat)
    
    @property
    def _flat(self):
        return Flat if isinstance(self, Immutable) else mFlat

    
class Sequentiable(ABC):
    '''Base class for all objects that can be iterated'''
    
    def __init__(self, *args):
        '''Directly called upon instantiation of concrete subtypes. (e.g.: at
        Vec[2, float](1, 2) rather than Vec(1, 2)).'''

        dtype = self.dtype
        if dtype is None:
            self.flat = self._flat(args)
        else:
            self.flat = self._flat([convert(x, dtype) for x in args], False)
            
    def __str__(self):
        name = type(self).__name__
        data = ', '.join(map(repr, self))
        return '%s(%s)' % (name, data)

    def __repr__(self):
        name = type(self).__origin__.__name__
        data = ', '.join(map(repr, self))
        return '%s(%s)' % (name, data)

    def __len__(self):
        return len(self.flat)

    def __iter__(self):
        return iter(self.flat)

    def __getitem__(self, idx):
        return self.flat[idx]

    def __eq__(self, other):
        if self.shape != _shape(other):
            return False

        if self.__origin__ is getattr(other, '__origin__', None):
            return all(x == y for (x, y) in zip(self.flat, other.flat))
        else:
            return all(x == y for (x, y) in zip(self, other))

    @classmethod
    def fromdata(cls, data):
        '''Initializes from a sequence of values'''

        return cls(*data)

    
class SmallVectorsMeta(ParametricMeta):
    pass

class SmallVectorsBase(Serializable, Sequentiable, metaclass=SmallVectorsMeta):
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

    # Mathematical functions
    _sqrt = math.sqrt
    _sin = math.sin
    _cos = math.cos
    _tan = math.tan
    _acos = math.acos
    _asin = math.asin
    _atan = math.atan
    _atan2 = math.atan2
    _number = (float, int, Number)

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
        
        if cls.__parameters__ is None or cls.shape is None:
            default_ns = _default_smallvectors_type_ns(cls.__parameters__)
            for k, v in default_ns.items():
                if getattr(cls, k, None) is None:
                    setattr(cls, k, v)
        
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

    def convert(self, dtype):
        '''Convert object to the given data type'''

        cls = type(self)
        if dtype is self.dtype:
            return self
        else:
            return cls.__origin__(*self, dtype=dtype)
