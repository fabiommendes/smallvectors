'''
==================================
Metaclasses for mathematical types
==================================

The smallvectors package makes an extensive use of metaclasses that probably
needs some explanation. First, many types can be parametrized generating
different subtypes. For instance, ``Vec`` is an abstract type and the
concrete incarnations are created using ``Vec[N, T]``, for the number of
components N and element type T. In order for this to work, we need a
metaclass that overrides __getitem__ in order to implement the correct
behavior.

A lot of this functionality can be shared between most of smallvector types
such as Vec, Point, Mat, Affine, Quaternion, etc. However, each of these base
abstract types must store some internal state that is used to manage creation
of new classes, cache, some dimension specific properties, etc. All these types
inherit from BaseAbstractType.

The concrete types, however, do not inherit from the corresponding abstract
type. Although ``issubclass(Vec[2, float], Vec) is True``, Vec is not in the
``mro()`` of any concrete type ``Vec[N, T]``. In fact, ``Vec`` is a factory for
all derived ``Vec[N, T]`` types. It is somewhat similar to a metaclass but we
did not want an implementation that requires talking about meta-meta-classes.
'''
import six
import abc
import math
import numbers
import types
import operator as op
from operator import mul
from functools import reduce
from generic import promote_type, convert, GenericObject
from generic.operator import add
from . import Flat, mFlat, privateclassmethod, binop_factory
from . import shape as get_shape, dtype as get_dtype


class BaseAbstractMeta(abc.ABCMeta):

    '''
    Meta class for BaseAbstractType.
.

    It overrides __getitem__ to implement type parametrization and __call__
    to prevent creating instances of the abstract type and in order to find a
    suitable concrete subtype from the type of the arguments.
    '''

    def __new__(cls, name, bases, ns):
        new = abc.ABCMeta.__new__(cls, name, bases, ns)
        new.__subtypes__ = {}
        new.__by_parameters_bases__ = {}
        new.__by_shape_bases__ = {}
        new.__by_dtype_bases__ = {}
        return new

    def __call__(self, *args, **kwds):
        return self.__abstract_new__(*args, **kwds)

    def __getitem__(self, args):
        try:
            return self.__subtypes__[args]
        except KeyError:
            new = self.subtype_new(args)
            self.__subtypes__[args] = new
            self.__subtypes__[args] = new = self.subtype_decorate(new)
            self.register(new)
            return new


@six.add_metaclass(BaseAbstractMeta)
class ABC(object):
    """Helper class that provides a standard way to create an ABC using
    inheritance.
    """


class BaseAbstractType(ABC):

    '''
    All abstract mathematical types such as Vec, Mat, Affine, etc are
    subclasses of this.
    '''
    #
    # Default value for class parameters
    #
    ndim = None
    size = None
    shape = None
    parameters = None
    dtype = None
    __root__ = None

    #
    # Overridable math functions. Can be useful when implementing types that
    # uses non-standard numeric types such as decimals or sympy values.
    #
    __slots__ = ()
    __private_names__ = {
        '__class__',
        '__subtypes__',
        '__by_parameters_bases__',
        '__by_shape_bases__',
        '__by_dtype_bases__',
    }
    _null_value = None
    _sqrt = math.sqrt
    _sin = math.sin
    _cos = math.cos
    _tan = math.tan
    _acos = math.acos
    _asin = math.asin
    _atan = math.atan
    _atan2 = math.atan2
    _fast_number = (float, int)
    _number = (float, int, numbers.Number)
    # _flat = None

    @privateclassmethod
    def split_parameters(cls, parameters, force_dtype=True):  # @NoSelf
        '''Split the parameters into shape and dtype components.'''

        return parameters[:-1], parameters[-1]

    @privateclassmethod
    def subtype_name(cls, parameters):  # @NoSelf
        '''Compute the subtype's name from its parameters.'''

        shape, dtype = cls.split_parameters(parameters)
        args = ', '.join(map(str, shape)) + ', ' + dtype.__name__
        return '%s[%s]' % (cls.__name__, args)

    @privateclassmethod
    def subtype_bases(cls, parameters):  # @NoSelf
        '''Compute the base class from parameters'''

        # Try parameters/shapes/dtype
        try:
            return cls.__by_parameters_bases__[parameters]
        except KeyError:
            shape, dtype = cls.split_parameters(parameters)
        try:
            return (cls.__by_shape_bases__.get(shape) or
                    cls.__by_dtype_bases__[dtype])
        except KeyError:
            return (GenericObject,)

    @privateclassmethod
    def subtype_namespace(cls, parameters):  # @NoSelf
        '''Compute the namespace for a new type based on its parameters'''

        D = {}
        D.update(cls.subtype_methods(parameters))
        D.update(cls.subtype_constants(parameters))
        return D

    @privateclassmethod
    def subtype_constants(cls, parameters):  # @NoSelf
        '''Return the class constants such as shape, dtype, ndim, which are
        directely based on the type's parameters.'''

        shape, dtype = cls.split_parameters(parameters)
        return {
            'dtype': dtype,
            'shape': shape,
            'size': reduce(mul, shape, 1),
            'ndim': len(shape),
            'parameters': parameters,
            '__name__': cls.subtype_name(parameters),
            '__root__': cls,
        }

    @privateclassmethod
    def subtype_methods(cls, parameters):  # @NoSelf
        '''Gather a list of subtype's methods and other constants that should
        be included in its namespace.

        By default, it gathers all methods and constants of a subclass that
        are not explicitly marked for exclusion.'''

        private = cls.__private_names__
        methods = {}
        abc_methods = {k: getattr(ABC, k) for k in dir(ABC)}

        for name in dir(cls):
            attr = getattr(cls, name)

            # Blacklist private methods and attributes
            if name in private or getattr(attr, 'private', False):
                continue

            # Avoid any methods from the ABC base
            if name in abc_methods and abc_methods[name] == attr:
                continue

            # Conditional insertion
            if hasattr(attr, 'cond_function'):
                if not attr.cond_function(parameters):
                    continue

            # Rebind classmethods
            if six.PY3 and isinstance(attr, types.MethodType):
                attr = classmethod(attr.__func__)

            if six.PY3:
                methods[name] = attr
            else:
                if isinstance(attr, types.MethodType):
                    if attr.im_self is None:
                        attr = attr.im_func
                    else:
                        attr = classmethod(attr.im_func)

                methods[name] = attr

        if methods.get('__slots__', ()) == ():
            if 'flat' not in methods:
                methods['__slots__'] = ('flat',)
            else:
                methods.pop('__slots__', None)

        return methods

    @privateclassmethod
    def subtype_new(cls, parameters, bases=None):  # @NoSelf
        '''Creates a new subtype from the given parameters.'''

        if bases is None:
            bases = cls.subtype_bases(parameters)
        name = cls.subtype_name(parameters)
        namespace = cls.subtype_namespace(parameters)
        new = type(name, bases, namespace)

        # Remove methods that appear in all bases bases up to GenericObject
        D = new.__dict__
        no_deletion = set([
            '__class__', '__name__', '__module__', '__dict__', '__doc__',
            '__subclasshook__', '__hash__', '__repr__', '__eq__',  # eq???
            'dtype', 'shape', 'size', 'parameters', ])

        for name in list(D):
            if name in no_deletion:
                continue

            for cls in new.mro()[1:]:
                if cls in [GenericObject, object]:
                    break
                if hasattr(cls, name):
                    delattr(new, name)
                    value = getattr(new, name)
                    value2 = getattr(object, name, None)
                    break
        return new

    @privateclassmethod
    def subtype_decorate(cls, new):  # @NoSelf
        '''Post-process subtype after inclusion in the class dictionary.

        This may be useful for subtypes that may spawn the creation of other
        subtypes.'''

        return new

    @privateclassmethod
    def subtype_register_base(cls, base=None,  # @NoSelf
                              parameters=None, shape=None, dtype=None):
        '''Register a base class for the given parameters or shape or dtype.

        This class will be preferred over the default when creating new
        subtypes. A parameter specification has the highest priority followed
        by shape and dtype.'''

        if base is None:
            def decorator(base):
                cls.subtype_register_base(base, parameters, shape, dtype)
                return base
            return decorator

        if parameters is not None:
            cls.__by_parameters_bases__[parameters] = (base,)
        if shape is not None:
            cls.__by_shape_bases__[shape] = (base,)
        if dtype is not None:
            cls.__by_dtype_bases__[dtype] = (base,)

    #
    # Common methods
    #
    @classmethod
    def __abstract_new__(cls, *args, shape=None, dtype=None):  # @NoSelf
        '''
        This function is called when user tries to instatiate an abstract
        type. It just finds the proper concrete type and instantiate it.
        '''
        if dtype is None:
            dtype = get_dtype(args)
        if shape is None:
            shape = get_shape(args)
        return cls[shape + (dtype,)](*args)

    def __init__(self, *args):
        '''Directly called upon instantiation of concrete subtypes. (e.g.: at
        Vec[2, float](1, 2) rather than Vec(1, 2)).'''

        dtype = self.dtype
        self.flat = self._flat([convert(x, dtype) for x in args], False)

    #
    # Alternate constructors
    #
    @classmethod
    def from_flat(cls, flat, copy=True):
        '''Instantiate object from flat data'''

        dtype = cls.dtype
        new = cls.__new__(cls)
        if copy:
            new.flat = cls._flat([convert(x, dtype) for x in flat], False)
        else:
            new.flat = cls._flat(flat, False)
        return new

    @classmethod
    def from_data(cls, data):
        '''Initializes from a sequence of values'''

        return cls(*data)

    @classmethod
    def null(cls, shape=None):
        '''Return an object in which all components are zero'''

        if cls._null_value is not None:
            null = cls._null_value
        else:
            null = convert(0, cls.dtype)
        return cls.from_flat([null] * cls.size, shape=shape)

    #
    # Object conversion
    #
    def convert(self, dtype):
        '''Convert object to the given data type'''

        cls = type(self)
        if dtype is self.dtype:
            return self
        else:
            return cls.__root__(*self, dtype=dtype)

    #
    # Investigate object's properties
    #
    def is_null(self):
        '''Checks if object has only null components'''

        return all(x == 0.0 for x in self.flat)

    def is_unity(self, tol=1e-6, norm=None):
        '''Return True if the norm equals one within the given tolerance.

        Object must define a normalization method `norm`.'''

        value = self.norm(norm)
        return abs(value - 1) < tol

    def norm(self, which=None):
        '''Returns the norm of an object.
        
        The `which` attribute controls the method used to compute the norm.'''

        tname = type(self).__name__
        raise TypeError('%s object does not define a norm' % tname)

    def norm_sqr(self, which=None):
        '''Returns the squared norm of object.'''

        value = self.norm(which)
        return value * value

    def normalized(self, which=None):
        '''Return a normalized version of object.

        Normalizes according to the given method.'''

        try:
            Z = self.norm(which)
            return self / Z
        except ZeroDivisionError:
            return self

    #
    # Python protocols
    #
    def __len__(self):
        return len(self.flat)

    def __iter__(self):
        return iter(self.flat)

    def __getitem__(self, idx):
        return self.flat[idx]

    def __repr__(self):
        name = type(self).__name__
        data = ', '.join(map(str, self.flat))
        return '%s(%s)' % (name, data)

    def __nonzero__(self):
        return bool(self.size)

    def __abs__(self):
        return self.__norm(None)

    #
    # Arithmetic operations
    #
    ADD_ELEMENTWISE = True
    MUL_ELEMENTWISE = True
    ADD_SCALAR = True
    MUL_SCALAR = True

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    __add__, __radd2__ = binop_factory(op.add)
    __sub__, __rsub2__ = binop_factory(op.sub)
    __mul__, __rmul__ = binop_factory(op.mul)
    __truediv__, __rtruediv__ = binop_factory(op.truediv)
    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __neg__(self):
        return self.from_flat([-x for x in self], copy=False)

    #
    # Relations
    #
    def __eq__(self, other):
        if self.shape != get_shape(other):
            return False

        if self.__root__ is getattr(other, '__root__', None):
            return all(x == y for (x, y) in zip(self.flat, other.flat))
        else:
            return all(x == y for (x, y) in zip(self, other))


class Mutable(BaseAbstractType):
    '''Base class for all mutable types'''

    __slots__ = ()

    _flat = mFlat
    
    def normalize(self, which=None):
        '''Normalizes object *INPLACE* using the given method.'''

        self /= self.norm(which)


class Immutable(BaseAbstractType):
    '''Base class for all immutable types'''

    __slots__ = ()

    _flat = Flat

    def __hash__(self):
        return hash(tuple(self))


@add.overload([BaseAbstractType, BaseAbstractType])
def add_base(x, y):
    if x.__root__ is y.__root__:
        dtype = promote_type(x.dtype, y.dtype)
        return x.convert(dtype) + y.convert(dtype)
    else:
        fmt = type(x).__name__, type(y).__name__
        raise TypeError('invalid operation: add(%s, %s)' % fmt)
