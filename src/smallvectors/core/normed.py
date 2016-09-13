import abc

from smallvectors.core.mixins import MathFunctionsMixin
from smallvectors.core.util import mutating
from smallvectors.core.mutability import Immutable
from smallvectors.core import ABC


class Normed(MathFunctionsMixin, ABC):
    """
    Base class for all objects that have a notion of norm
    """

    __slots__ = ()

    def __abs__(self):
        return self.norm(None)

    @abc.abstractmethod
    def norm(self, which=None):
        """
        Returns the norm of an object.

        The `which` attribute controls the method used to compute the norm.
        """

        tname = type(self).__name__
        raise TypeError('%s object does not define a norm' % tname)

    def norm_sqr(self, which=None):
        """
        Returns the squared norm of an object using the desired metric.

        If object does not define norm_sqr, it tries to compute the norm using
        obj.norm() and squaring it.
        """

        value = self.norm(which)
        return value * value

    def normalize(self, which=None):
        """
        Return a normalize version of object.

        Normalizes according to the given method.
        """

        try:
            Z = self.norm(which)
            return self / Z
        except ZeroDivisionError:
            raise ValueError('null element cannot be normalized')

    @mutating
    def inormalize(self, which=None):
        """
        Normalizes object *INPLACE*.
        """

        if isinstance(self, Immutable):
            raise TypeError('cannot normalize immutable object')

        self.__idiv__(self.norm(which))

    def is_unity(self, norm=None, tol=1e-6):
        """
        Return True if the norm equals one within the given tolerance.

        Object must define a normalization method `norm`.
        """

        value = self.norm(norm)
        return abs(value - 1) < tol

    def is_null(self):
        """
        Return true if object has a null norm.
        """

        return self.norm() == 0.0

    def __idiv__(self, param):
        return NotImplementedError


class InnerProduct(Normed):
    """
    Object that implements an inner product with the .dot() method.
    """

    def dot(self, other):
        """
        Dot (inner) product with 'other' object.
        """

        raise NotImplementedError('subclasses must implement the .dot() '
                                  'method')

    def norm(self, which=None):
        if which is None or which == 'euclidean':
            return self._sqrt(self.dot(self))
        else:
            return super().norm(which)

    def norm_sqr(self, which=None):
        if which is None or which == 'euclidean':
            return self.dot(self)
        else:
            return super().norm_sqr(which)