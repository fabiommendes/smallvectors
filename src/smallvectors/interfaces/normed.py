import abc


class Normed(metaclass=abc.ABCMeta):
    """
    Base class for all objects that have a notion of norm
    """

    __slots__ = ()

    def __abs__(self):
        return self.norm()

    @abc.abstractmethod
    def dot(self, other):
        """
        Dot (inner) product with 'other' object.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def norm(self, norm=None):
        """
        Returns the norm of an object.

        The `which` attribute controls the method used to compute the norm.
        """

        tname = type(self).__name__
        raise TypeError('%s object does not define a norm' % tname)

    def norm_sqr(self, norm=None):
        """
        Returns the squared norm of an object using the desired metric.

        If object does not define norm_sqr, it tries to compute the norm using
        obj.norm() and squaring it.
        """

        value = self.norm(norm)
        return value * value

    def normalized(self, norm=None):
        """
        Return a normalize version of object.

        Normalizes according to the given method.
        """

        try:
            Z = self.norm(norm)
            return self / Z
        except ZeroDivisionError:
            raise ValueError('null element cannot be normalized')

    def is_unity(self, norm=None, tol=1e-6):
        """
        Return True if the norm equals one within the given tolerance.

        Object must define a normalization method `norm`.
        """

        value = self.norm(norm)
        return abs(value - 1) <= tol

    def is_null(self, tol=0.0):
        """
        Return true if object has a null norm.
        """

        return abs(self.norm() - tol) <= tol
