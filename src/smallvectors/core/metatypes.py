'''
==================================
Metaclasses for mathematical types
==================================

The ``smallvectors`` package puts objects into a hierarchy...



'''
import six

###############################################################################
#                            Metaclass types
###############################################################################
#
# The smallvectors package makes an extensive use of metaclasses that probably
# needs some explanation. First, many types can be parametrized generating
# different subtypes. For instance, ``Vec`` is an abstract type and the
# concrete incarnations are created using ``Vec[N, T]``, for the number of
# components N and element type T. In order for this to work, we need a
# ``VecMeta`` class that overrides __getitem__ in order to implement the
# correct behavior.
#
# A lot of this functionality can be shared between most of smallvector types
# such as Vec, Point, Mat, Affine, Quaternion, etc. However, each of these
# base abstract types must store some internal state that is used to manage
# creation of new classes, cache, some dimension specific properties, etc.
# There are many ways to implement this. We decided to make them an instance of
# yet another type, this is a meta-metaclass!
#

BaseMeta = None


class MetaMeta(type):

    '''The meta meta class ensures that metaclasses are initialized properly'''

    def __new__(cls, name, bases, namespace):
        # Check if metaclass is sane
        if len(bases) > 1:
            raise ValueError('multiple inheritance is not allowed')
        if not issubclass(bases[0], type):
            raise ValueError('metaclass must be a "type" subclass')
        if not name.endswith('Meta'):
            raise NameError('metaclass name must end with Meta')

        # Create new metatype
        new = type.__new__(cls, name, bases, namespace)

        # Check if it has the necessary attributes and return
        if BaseMeta in bases and new.ndims is None:
            raise ValueError('ndims class attribute must be specified')

        return new


@six.add_metaclass(MetaMeta)
class BaseMeta(type):

    '''VecMeta, MatMeta, etc derive from this type. Instances of BaseMeta
    define concrete parametrized types in the smallvectors package type
    hierarchy.

    BaseMeta implements the __getitem__ functionality that allows root types to
    be subscriptable. VecMeta thus is the metaclass for the abstract Vec type.
    Vec[N, T] calls the __getitem__ method defined in the VecMeta class and
    produces a subclass of Vec that is a concrete type which can be
    instantiated. Upon creation of a new subtype, some type attributes are set
    to the correct value.
    '''

    def __new__(cls, name, bases, namespace):
        root = type.__new__(cls, name, bases, namespace)

        # Save some attributes to the root element
        root.ndims = cls.ndims
        root.__root__ = root
        root.__subtypes__ = {}
        return root

    #
    # Attributes
    #
    ndims = None
    root = None
    shape = None
    dtype = None
    is_root = True

    @property
    def argnames(self):
        names = [str(x) for x in self.shape]
        names.append(self.dtype.__name__)
        return tuple(names)

    @property
    def size(self):
        return multiply(self.shape)

    @property
    def root_name(self):
        return self.__root__.__name__

    @property
    def full_name(self):
        return '%s[%s]' % (self.root_name, ', '.join(self.argnames))

    #
    # Methods
    #
    def new_subtype(self, shape, dtype, bases=()):
        '''Create a new parametric type with the given arguments without fully
        initializing it'''

        if not self.is_root:
            raise ValueError('only the root type can be parametrized')
        root = self
        cache = root.__subtypes__

        try:
            return cache[shape, dtype]
        except KeyError:
            # Normalize arguments
            shape = tuple(shape)
            dtype = dtype or object
            if not all(isinstance(x, int) for x in shape):
                raise ValueError('shape sizes must be integers')
            if not isinstance(dtype, type):
                tname = type(dtype).__name__
                raise ValueError('expect a type, got %s' % tname)

            name = root.__name__
            bases += (root,)
            namespace = {}
            namespace.update(
                shape=shape,
                size=multiply(shape),
                dtype=dtype,
                root=root,
                is_root=False,
            )

            # Save in cache and return
            subtype = type.__new__(type(root), name, bases, namespace)
            cache[shape, dtype] = subtype
            subtype.__name__ = subtype.full_name
            return subtype

    def __getitem__(self, idx):
        # Force tuple indexes
        if not isinstance(idx, tuple):
            return self[(idx,)]

        # Check by the number of arguments if dtype was given
        if len(idx) == self.ndims:
            shape = idx
            dtype = object
        elif len(idx) == self.ndims + 1:
            shape = idx[:-1]
            dtype = idx[-1]
        else:
            N = self.ndims
            M = len(idx)
            raise TypeError('invalid number of type parameters: expected %s '
                            'or %s arguments, got %s' % (N, N + 1, M))

        try:
            return self.__subtypes__[shape, dtype]
        except KeyError:
            return self.new_subtype(shape, dtype)

    def __repr__(self):
        base = super(BaseMeta, self).__repr__()
        if self.is_root:
            return base
        else:
            qualname = '%s.%s' % (self.__module__, self.__qualname__)
            return "<class '%s[%s]'>" % (qualname, ', '.join(self.argnames))

    def __str__(self):
        return repr(self)


###############################################################################
#                         Metaclasses for specific types
###############################################################################
#
# Mixin classes
#
class VecOrPointMeta(BaseMeta):

    '''Common functionally for VecMeta and mVecMeta types.'''

    ndims = 1

    # TODO: convert to lazy properties
    #
    # These variables were defined for late binding. We shall assign the
    # correct values when the Vec, Point and Direction root classes exist
    #
    _vec_root = None
    _mvec_root = None
    _direction_root = None
    _point_root = None
    _mpoint_root = None

    def __init_subtype__(self, new):
        args = new.__args__
        new._vector_type = self._vec_root.__new_subtype__(args)
        new._vector_from_flat = new._vector_type.from_flat
        new._direction_type = self._direction_root.__new_subtype__(args)
        new._direction_from_flat = new._direction_type.from_flat
        new._point_type = self._point_root.__new_subtype__(args)
        new._point_from_flat = new._point_type.from_flat


class MatMeta(BaseMeta):

    '''Metaclass for the Mat class'''

    ndims = 2


###############################################################################
#                         Utility functions
###############################################################################
def multiply(args):
    '''Return the product of a list of arguments'''

    prod = 1
    for x in args:
        prod *= x
    return prod
