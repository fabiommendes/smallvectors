from generic import promote_type
from generic.op import Object
from generic.parametric import ParametricMeta, Mutable as _Mutable, Any

from smallvectors.core.flat import Flat, mFlat, Flatable
from smallvectors.core.mixins import MathFunctionsMixin
from smallvectors.core.sequentiable import Sequentiable
from smallvectors.core.util import default_smallvectors_type_ns
from smallvectors.utils import dtype as _dtype, shape as _shape


# noinspection PyUnresolvedReferences
# noinspection PyAbstractClass
class SmallVectorsBase(MathFunctionsMixin, Flatable, Sequentiable, Object,
                       metaclass=ParametricMeta):
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
        """
        Return a copy of object.
        """

        return NotImplemented


