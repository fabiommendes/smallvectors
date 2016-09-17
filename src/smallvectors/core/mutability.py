import sys

from generic import parametric


class complementary_descriptor:
    def __init__(self, attr):
        self.attr = attr

    def __get__(self, obj, cls):
        try:
            return cls.__dict__[self.attr]
        except KeyError:
            other = get_complementary(cls)
            setattr(cls, self.attr, other)
            return other


class setter_descriptor:
    def __init__(self, attr):
        self.attr = attr

    def __get__(self, obj, cls):
        try:
            return cls.__dict__[self.attr]
        except KeyError:
            setattr(cls, self.attr, cls)
            return cls


class MutabilityAPI:
    """
    Common API for Mutable and Immutable classes.
    """

    __slots__ = ()

    def __getstate__(self):
        raise NotImplementedError

    def is_mutable(self):
        raise NotImplementedError

    def is_immutable(self):
        raise NotImplementedError

    def mutable(self):
        raise NotImplementedError

    def immutable(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class Mutable(parametric.Mutable, MutabilityAPI):
    """
    Base class for all mutable types.
    """

    __slots__ = ()
    __immutable_class__ = complementary_descriptor('__immutable_class')
    __mutable_class__ = setter_descriptor('__mutable_class')

    def __getstate__(self):
        return NotImplemented

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

        cls = self.__immutable_class__
        return cls(*self.__getstate__())

    def copy(self):
        return NotImplemented


class Immutable(parametric.Immutable, MutabilityAPI):
    """
    Base class for all immutable types.
    """

    __slots__ = ()
    __mutable_class__ = complementary_descriptor('__mutable_class')
    __immutable_class__ = setter_descriptor('__immutable_class')

    def __getstate__(self):
        return NotImplemented

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

        cls = self.__mutable_class__
        return cls(*self.__getstate__())

    def immutable(self):
        """
        Return an immutable copy of object.
        """

        return self

    def copy(self):
        return self


def get_complementary(cls):
    """
    Return mutable from immutable and vice-versa.
    """

    # Get origin type and its name
    if cls.__origin__ is None:
        origin = cls
    else:
        origin = cls.__origin__
    name = cls.__name__
    origin_name = origin.__name__

    # Check if the mutability naming convention is followed. If so, compute the
    # complementary origin name. If not, raise an attribute error.
    is_mutable = issubclass(cls, Mutable)
    if is_mutable and not origin_name.startswith('m'):
        raise AttributeError('no immutable class associated with %s' % name)
    elif is_mutable:
        complementary_name = origin_name[1:]
    else:
        complementary_name = 'm' + origin_name

    # We check for the complementary class in the module that defines the origin
    # class
    mod = sys.modules[origin.__module__]
    complementary_origin = getattr(mod, complementary_name)

    # Apply parameters, if necessary
    if cls is origin:
        return complementary_origin
    else:
        return complementary_origin[cls.__parameters__]
