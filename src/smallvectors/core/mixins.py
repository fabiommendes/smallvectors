import cmath
import math
from numbers import Number

from smallvectors.core import ABC


class MathFunctionsMixin(ABC):
    """
    Defines a set of overrides for mathematical functions as private methods.

    Class implementors should use these methods instead of functions in the
    math module so subclasses can reuse implementations with their own math
    functions (maybe using numpy, sympy, cmath, etc).
    """

    __slots__ = ()

    # Mathematical functions
    _sqrt = math.sqrt
    _sin = math.sin
    _cos = math.cos
    _tan = math.tan
    _acos = math.acos
    _asin = math.asin
    _atan = math.atan
    _atan2 = math.atan2
    _float = float
    _floating = float
    _number = (float, int, Number)

    # Smallvector types
    _vec = None
    _mvec = None
    _point = None
    _mpoint = None
    _direction = None
    _mat = None
    _mmat = None
    _rotmat = None
    _affine = None
    _identity = None


class ComplexMathFunctionsMixin(MathFunctionsMixin):
    """
    Mixin that uses functions from the cmath module.
    """

    _sqrt = cmath.sqrt
    _sin = cmath.sin
    _cos = cmath.cos
    _tan = cmath.tan
    _acos = cmath.acos
    _asin = cmath.asin
    _atan = cmath.atan
