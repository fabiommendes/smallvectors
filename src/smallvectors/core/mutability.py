import sys
import types
from abc import abstractmethod
from functools import lru_cache

from generic import parametric


class complementary_mutable_class_descriptor:
    def __init__(self, attr):
        self.attr = attr

    def __get__(self, obj, cls):
        try:
            return cls.__dict__[self.attr]
        except KeyError:
            other = get_complementary_by_mutability(cls)
            setattr(cls, self.attr, other)
            return other


class auto_class_setter_descriptor:
    def __init__(self, attr):
        self.attr = attr

    def __get__(self, obj, cls):
        try:
            return cls.__dict__[self.attr]
        except KeyError:
            setattr(cls, self.attr, cls)
            return cls


@lru_cache(maxsize=256)
def get_slots(cls):
    members = (getattr(cls, name, None) for name in dir(cls))
    return [x for x in members if isinstance(x, types.MemberDescriptorType)]



class Mutable(parametric.ABC):
    """
    Base class for all mutable types.
    """

    __slots__ = ()
    __immutable_class__ = complementary_mutable_class_descriptor(
        '__immutable_class')
    __mutable_class__ = auto_class_setter_descriptor('__mutable_class')

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

        new = object.__new__(self.__immutable_class__)
        new.__setstate__(self.__getstate__())
        return new

    @abstractmethod
    def copy(self):
        """
        Return a copy.
        """

        return NotImplemented


class Immutable(parametric.ABC):
    """
    Base class for all immutable types.
    """

    __slots__ = ()
    __mutable_class__ = complementary_mutable_class_descriptor(
        '__mutable_class')
    __immutable_class__ = auto_class_setter_descriptor('__immutable_class')

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

        new = object.__new__(self.__immutable_class__)
        new.__setstate__(self.__getstate__())
        return new

    def immutable(self):
        """
        Return an immutable copy of object.
        """

        return self

    def copy(self):
        """
        Immutable objets do not have to be copied.
        """

        return self


def get_complementary_by_mutability(cls):
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


#
# Register classes
#
class MutabilityAPI(parametric.ABC):
    """
    Abstract class that defines the MutabilityAPI interface.
    """

    # We use monkey-patching to reduce one degree in the class hierarchy
    def __getstate__(self):
        cls = self.__class__
        return [slot.__get__(self, cls) for slot in get_slots(cls)]

    def __setstate__(self, state):
        cls = type(self.__class__)
        for slot, value in zip(get_slots(cls), state):
            slot.__set__(self, value)

Mutable.__getstate__ = MutabilityAPI.__getstate__
Immutable.__getstate__ = MutabilityAPI.__getstate__
Mutable.__setstate__ = MutabilityAPI.__setstate__
Immutable.__setstate__ = MutabilityAPI.__setstate__
MutabilityAPI.register(Mutable)
MutabilityAPI.register(Immutable)
parametric.Mutable.register(Mutable)
parametric.Immutable.register(Immutable)
