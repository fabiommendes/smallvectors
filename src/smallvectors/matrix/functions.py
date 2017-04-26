from smallvectors.matrix.mat import Mat


def asmatrix(m):
    """
    Return object as an immutable matrix.
    """

    if isinstance(m, Mat):
        return m
    else:
        return Mat(*m)


def identity(N, dtype=float):
    """
    Return an identity matrix of size N by N.
    """

    return Mat.from_diag([1] * N)