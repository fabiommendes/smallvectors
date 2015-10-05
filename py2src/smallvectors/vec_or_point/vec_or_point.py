from __future__ import division
from generic.operator import truediv_safe
from ..core import BaseAbstractType, privateclassmethod


class VecOrPoint(BaseAbstractType):

    '''Common implementations for Vec and Point types'''

    #
    # Class constants
    #
    from smallvectors.core.util import dtype as __dtype
    __dtype = staticmethod(__dtype)
    ndim = 1
    ADD_SCALAR = False
    MUL_ELEMENTWISE = False
    __COORDINATE_NAMES = {0: 'x', 1: 'y', 2: 'z', 3: 'w'}
    __PROPERTIES = {}
    __slots__ = ()

    #
    # Factory functions (Attribute access)
    #
    @privateclassmethod
    def __coordinate_property(cls, idx):  # @NoSelf
        try:
            prop = cls.__PROPERTIES[idx]

        except KeyError:
            @property
            def prop(self):
                return self[idx]

            @prop.setter
            def prop(self, value):
                try:
                    self[idx] = value
                except:
                    raise AttributeError('property is not writable')

            cls.__PROPERTIES[idx] = prop

        return prop

    @privateclassmethod
    def subtype_methods(cls, parameters):  # @NoSelf
        methods = super(VecOrPoint, cls).subtype_methods(parameters)

        # Get property names
        (size,), _dtype = cls.split_parameters(parameters)
        for idx in range(size):
            prop = cls.__coordinate_property(idx)
            try:
                methods[cls.__COORDINATE_NAMES[idx]] = prop
            except KeyError:
                pass
            methods['x%s' % idx] = prop

        return methods

    @classmethod
    def __abstract_new__(cls, *args, **_3to2kwargs):  # @NoSelf
        if 'dtype' in _3to2kwargs: dtype = _3to2kwargs['dtype']; del _3to2kwargs['dtype']
        else: dtype = None
        '''
        This function is called when user tries to instatiate an abstract
        type. It just finds the proper concrete type and instantiate it.
        '''
        if dtype is None:
            dtype = cls.__dtype(args)
        return cls[len(args), dtype](*args)

    #
    # Representation and conversions
    #
    def as_vector(self):
        '''Returns a copy of object as a vector'''

        return self.__vector__(*self)

    def as_direction(self):
        '''Returns a normalized copy of object as a direction'''

        return self.__direction__(*self)

    def as_point(self):
        '''Returns a copy of object as a point'''

        return self.__point__(*self)

    def copy(self, x=None, y=None, z=None, w=None, **kwds):
        '''Return a copy possibly overriding some components'''

        data = list(self)
        if x is not None:
            data[0] = x
        if y is not None:
            data[1] = y
        if z is not None:
            data[2] = z
        if w is not None:
            data[3] = w

        # Keywords of the form e1=foo, e2=bar, etc
        for k, v in kwds.items():
            if not k.startswith('x'):
                raise TypeError('invalid argument %r' % k)
            data[int(k[1:]) + 1] = v

        return self.from_flat(data)

    #
    # Geometric properties
    #
    def almost_equal(self, other, tol=1e-3):
        '''Return True if two smallvectors are almost equal to each other'''

        return (self - other).norm_sqr() < tol * tol

    def distance(self, other):
        '''Computes the distance between two objects'''

        if len(self) != len(other):
            N, M = len(self), len(other)
            raise IndexError('invalid dimensions: %s and %s' % (N, M))

        deltas = (x - y for (x, y) in zip(self, other))
        return self._sqrt(sum(x * x for x in deltas))

    def lerp(self, other, weight):
        '''Linear interpolation between two objects.

        The weight attribute defines how close the result will be from the
        argument. A weight of 0.5 corresponds to the middle point between
        the two objects.'''

        if not 0 <= weight <= 1:
            raise ValueError('weight must be between 0 and 1')

        return (other - self) * weight + self

    def middle(self, other):
        '''The midpoint to `other`. The same as ``obj.lerp(other, 0.5)``'''

        return (self + other) / 2

    #
    # Arithmetical operations overrides
    #
    def __rtruediv__(self, other):
        return truediv_safe(other, self)
