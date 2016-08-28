def dot(u, v):
    """
    Return the dot product between the two objects.

    Example:

        >>> dot((1, 2), (3, 4))
        11
    """

    try:
        return u.dot(v)
    except AttributeError:
        pass

    try:
        return v.dot(u)
    except AttributeError:
        pass

    if len(u) == len(v):
        return sum(x * y for (x, y) in zip(u, v))
    else:
        raise ValueError('length mismatch: %s and %s')


def cross(v1, v2):
    """
    Return the cross product between two vectors.
    """

    x1, y1 = v1
    x2, y2 = v2
    return x1 * y2 - x2 * y1

