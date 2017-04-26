from .vec import Vec
from .vec_2d import Vec2
from .vec_3d import Vec3

VEC_DIMENSIONS_MAP = {2: Vec2, 3: Vec3}


def asvector(obj):
    """
    Return object as an immutable vector.
    """

    if isinstance(obj, Vec):
        return obj

    vec_class = VEC_DIMENSIONS_MAP[len(obj)]
    return vec_class(*obj)
