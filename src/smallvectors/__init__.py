from .__meta__ import __version__, __author__
from .core import *
from .functions import *
from .utils import *
from .vector import *
from .matrix import *
from .affine import *
from .vecarray import *

# Late binding of smallvectors types to MathFunctionsMixin
MathFunctionsMixin._vec = Vec
MathFunctionsMixin._mvec = mVec
MathFunctionsMixin._point = Point
MathFunctionsMixin._mpoint = mPoint
MathFunctionsMixin._mat = Mat
MathFunctionsMixin._mmat = mMat
MathFunctionsMixin._rotation2D = Rotation2d
MathFunctionsMixin._rotation3D = Rotation3d
MathFunctionsMixin._affine = Affine
