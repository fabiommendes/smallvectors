'''
A collection of useful decorators
'''
from generic import operator as generic_op
from .flatobject import Flat

#
# Generic decorators
#


class lazy(object):

    '''Mark an attribute to be lazy.'''

    def __init__(self, func):
        self._func = func

    def __get__(self, cls, obj=None):
        if obj is None:
            return self
        else:
            return self._func(obj)


#
# Method factories
#
def conditional_method(cond_function):
    '''Receives a boolean function of the parameters as argument. The method
    is inserted in a concrete class only if condition is satisfied for the
    given class.'''

    def decorator(func):
        func.cond_function = cond_function
        return func
    return decorator


def method_factory(func):
    pass


def default_implementation(func):
    pass


def privateclassmethod(func):
    func.private = True
    return classmethod(func)


def binop_factory(op, safe_generic=None, reverse=False,
                  is_add=False, is_mul=False):
    '''Creates a method for the given binary operator.

    Return a tuple with __op__ and __rop__ functions.'''

    from .util import shape
    make_flat = Flat

    # Check if it is an multiplication or addition operator
    if op.__name__ in ['add', 'sub']:
        is_add = True
    if op.__name__ in ['mul', 'truediv']:
        is_mul = True

    # Choose the correct safe generic function from operator name
    if safe_generic is None:
        name = op.__name__
        safe_generic = getattr(generic_op, name + '_safe')

    def method(self, other):
        self_t = type(self)
        other_t = type(other)

        # Element-wise operation with objects of same root type
        if (is_add and self.ADD_ELEMENTWISE or
                is_mul and self.MUL_ELEMENTWISE):

            if (self_t is other_t or
                    getattr(other_t, '__root__', None) is self_t.__root__):

                flat = make_flat(map(op, self.flat, other.flat))

                if isinstance(flat[0], self.dtype):
                    return self_t.from_flat(flat, False)
                else:
                    dtype = type(flat[0])
                    cls = self.__root__[self.shape + (dtype,)]
                    return cls.from_flat(flat, False)

        # Scalar operation with scalars
        if is_mul and self.MUL_SCALAR or is_add and self.ADD_SCALAR:
            if isinstance(other, self._number):
                data = [op(x, other) for x in self.flat]
                dtype = type(data[0])
                cls = self.__root__[self.shape + (dtype,)]
                return cls.from_flat(data, copy=False)

        return safe_generic(self, other)

    def rmethod(self, other):
        other_t = type(other)
        self_t = type(self)

        # Scalar operation with scalars
        if isinstance(other, self._number):
            if ((is_add and self_t.ADD_SCALAR) or
                    (is_mul and self_t.MUL_SCALAR)):
                data = [op(other, x) for x in self.flat]
                cls = self.__root__[self.parameters]
                return cls.from_flat(data, copy=False)
            else:
                fmt = op.__name__, self_t.__name__, other_t.__name__
                raise TypeError('do not support %s(%s, %s)' % fmt)

        # Elementwise operation with objects of different types
        elif ((is_add and self_t.ADD_ELEMENTWISE) or
              (is_mul and self_t.MUL_ELEMENTWISE)):
            if self.shape == getattr(other, 'shape', None):
                data = [op(x, y) for x, y in zip(other.flat, self.flat)]
                return self.__root__.from_flat(data)

            if self.shape == shape(other):
                data = [op(x, y) for x, y in zip(other, self.flat)]
                return self.__root__.from_flat(data)

    return method, rmethod
