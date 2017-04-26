from smallvectors.direction.direction import Direction2, Direction

DIRECTION_DIMENSIONS_MAP = {2: Direction2}


def asdirection(obj):
    """
    Return the argument as a Direction instance.
    """

    if isinstance(obj, Direction):
        return obj

    direction_class = DIRECTION_DIMENSIONS_MAP[len(obj)]
    return direction_class(*obj)