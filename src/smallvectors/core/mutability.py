from generic import parametric

from smallvectors.core.util import get_sibling_classes


class Mutable(parametric.Mutable):
    """
    Base class for all mutable types.
    """

    __slots__ = ()

    def __getstate__(self):
        return NotImplemented

    @property
    def __immutable_class__(self):
        cls = self.__class__
        mod = __import__(cls.__module__)
        immutable = getattr(mod, cls.__name__[1:])
        if not issubclass(immutable, Immutable):
            raise TypeError('%s should be immutable')
        setattr(cls, '__immutable_class__', immutable)
        return immutable

    @property
    def __mutable_class__(self):
        return self.__class__

    def is_mutable(self):
        """
        Return True if object is mutable.
        """

        return True

    def is_immutable(self):
        """
        Return True if object is immutable.
        """

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
            cls = self.__immutable_class__
        except AttributeError:
            siblings = get_sibling_classes(type(self))
            immutable = [T for T in siblings if issubclass(T, Immutable)]
            assert len(immutable) == 1
            cls = type(self).__immutable_class__ = immutable[0]
        return cls(*self.__getstate__())

    def copy(self):
        return NotImplemented


class Immutable(parametric.Immutable):
    """
    Base class for all immutable types.
    """

    __slots__ = ()

    def __getstate__(self):
        return NotImplemented

    @property
    def __mutable_class__(self):
        cls = self.__class__
        mod = __import__(cls.__module__)
        mutable = getattr(mod, 'm' + cls.__name__)
        if not issubclass(mutable, Mutable):
            raise TypeError('%s should be mutable')
        setattr(cls, '__mutable_class__', mutable)
        return mutable

    @property
    def __immutable_class__(self):
        return self.__class__

    def is_mutable(self):
        """
        Return True if object is mutable.
        """

        return False

    def is_immutable(self):
        """
        Return True if object is immutable.
        """

        return True

    def mutable(self):
        """
        Return a mutable copy of object.
        """

        try:
            cls = self.__mutable_class__
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
