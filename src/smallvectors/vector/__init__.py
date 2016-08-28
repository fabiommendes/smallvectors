DIMENSION_BASES = {}
from .core import *
from . import promotion as _promotion
from . import vec_nd as _vec_nd
from . import vec_2d as _vec_2d
from . import vec_3d as _vec_3d
from . import vec_4d as _vec_4d
from .direction import Direction
from .point import *

# Maps dimensions to additional bases
DIMENSION_BASES.update({
    0: _vec_nd.Vec0D,
    1: _vec_nd.Vec1D,
    2: _vec_2d.Vec2D,
    3: _vec_3d.Vec3D,
    4: _vec_4d.Vec4D,
})
