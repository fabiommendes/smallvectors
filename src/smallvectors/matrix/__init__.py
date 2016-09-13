from .base import Mat, mMat, MatAny, asmatrix, asmmatrix, asamatrix, \
    identity, midentity
from .rotmat import *
from .square import SquareMixin
from ..vector import VecAny as _VecAny

SquareMixin._mat = Mat
SquareMixin._mmat = mMat
SquareMixin._identity = identity
SquareMixin._rotmatrix = Mat
_VecAny._rotmatrix = Mat