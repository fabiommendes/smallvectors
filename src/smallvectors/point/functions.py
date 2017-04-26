from smallvectors.point.point import Point2, Point

POINT_DIMENSIONS_MAP = {2: Point2}


def aspoint(obj):
    """
    Return object as an immutable point.
    """

    if isinstance(obj, Point):
        return obj

    point_class = POINT_DIMENSIONS_MAP[len(obj)]
    return point_class(*obj)