from smallvectors import SmallVectorsBase, MulScalar, AddElementWise, \
    Mutable
from smallvectors.tools import dtype as _dtype
from smallvectors.vector import DIMENSION_BASES


# noinspection PyAbstractClass
class LinearAny(SmallVectorsBase, AddElementWise, MulScalar):
    """
    Common implementations for Vec and Point types
    """

    __slots__ = ()
    __parameters__ = (int, type)
    __abstract__ = True

    @classmethod
    def __preparebases__(cls, params):
        try:
            bases = (DIMENSION_BASES[params[0]], cls)
        except KeyError:
            bases = (cls,)
        return bases

    @classmethod
    def __preparenamespace__(cls, params):
        ns = SmallVectorsBase.__preparenamespace__(params)
        if isinstance(params[0], int) and params[0] > 4:
            ns.update(__slots__='flat')
        return ns

    @staticmethod
    def __finalizetype__(cls):
        SmallVectorsBase.__finalizetype__(cls)

        # Make all x, y, z, etc property accessors. Mutable objects have
        # read/write accessors, while immutable types have only the read
        # accessor.
        def getter(i):
            return lambda self: self[i]

        def setter(i):
            return lambda self, value: self.__setitem__(i, value)

        for i in range(cls.size or 0):
            prop = property(getter(i))
            if issubclass(cls, Mutable):
                prop = prop.setter(setter(i))

            attr = 'x%i' % i
            if not hasattr(cls, attr):
                setattr(cls, attr, prop)
            if i < 4:
                attr = ('x', 'y', 'z', 'w')[i]
                if not hasattr(cls, attr):
                    setattr(cls, attr, prop)

    @classmethod
    def __abstract_new__(cls, *args, dtype=None):  # @NoSelf
        """
        This function is called when user tries to instantiate an abstract
        type. It just finds the proper concrete type and instantiate it.
        """
        if dtype is None:
            dtype = _dtype(args)
        return cls[len(args), dtype](*args)

    def __getitem_simple__(self, idx):
        return self.flat[idx]

    def __vector__(self, v):
        return NotImplemented

    def __direction__(self, v):
        return NotImplemented

    def __point__(self, v):
        return NotImplemented

    # Representation and conversions
    def as_vector(self):
        """
        Returns a copy of object as a vector
        """

        return self.__vector__(*self)

    def as_direction(self):
        """
        Returns a normalized copy of object as a direction
        """

        return self.__direction__(*self)

    def as_point(self):
        """
        Returns a copy of object as a point
        """

        return self.__point__(*self)

    def copy(self, x=None, y=None, z=None, w=None, **kwds):
        """
        Return a copy possibly overriding some components
        """

        data = list(self)
        if x is not None:
            data[0] = x
        if y is not None:
            data[1] = y
        if z is not None:
            data[2] = z
        if w is not None:
            data[3] = w

        # Keywords of the form x1=foo, x2=bar, etc
        for k, v in kwds.items():
            if not k.startswith('x'):
                raise TypeError('invalid argument %r' % k)
            data[int(k[1:]) + 1] = v

        return self.from_flat(data)

    # Geometric properties
    def almost_equal(self, other, tol=1e-3):
        """
        Return True if two vectors are almost equal to each other.
        """

        return (self - other).norm_sqr() < tol * tol

    def distance(self, other):
        """
        Computes the distance between two objects.
        """

        if len(self) != len(other):
            N, M = len(self), len(other)
            raise IndexError('invalid dimensions: %s and %s' % (N, M))

        deltas = (x - y for (x, y) in zip(self, other))
        return self._sqrt(sum(x * x for x in deltas))

    def lerp(self, other, weight):
        """
        Linear interpolation between two objects.

        The weight attribute defines how close the result will be from the
        argument. A weight of 0.5 corresponds to the middle point between
        the two objects.
        """

        if not 0 <= weight <= 1:
            raise ValueError('weight must be between 0 and 1')

        return (other - self) * weight + self

    def middle(self, other):
        """
        The midpoint to `other`. The same as ``obj.lerp(other, 0.5)``
        """

        return (self + other) / 2

    def move(self, *args):
        """
        An alias to obj.displaced().
        """

        if len(args) == 1:
            args = args[0]
        return self + args
