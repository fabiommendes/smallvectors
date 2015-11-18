import numbers
from smallvectors.matrix import Mat


class Unitary(Mat):

    """Base class for all unitary matrices (e.g., rotation matrices).
    
    The conjugate of an unitary matrix is equal to its inverse. Unitary matrices
    are immutable since almost any modification to the matrix would break 
    unitarity. """

    __slots__ = ()


    @classmethod
    def __preprarebases__(cls, params):
        N, M, dtype = params
        if N != M:
            raise TypeError('unitary matrices must be squared')
        if issubclass(dtype, numbers.Integral):
            raise TypeError('cannot create integer valued unitary matrices') 
        return super().__preparebases__(params)

    def inv(self):
        return self.conjugate()

    def det(self):
        return 1.0


class Rotation2d(Unitary[2, 2, float]):

    """Rotation matrix in 2D"""

    __slots__ = ('theta',)

    def __init__(self, theta):
        self.theta = theta + 0.0
        C = self._cos(theta)
        S = self._sin(theta)
        super().__init__([C, -S], [S, C])

    def rotate(self, theta):
        return self.__class__(self.theta + theta)


class Rotation3d(Unitary[3, 3, float]):
    """Rotation matrix in 3D"""

    __slots__ = ('theta',)

    def __init__(self, theta, axis):
        self.theta = float(theta)
        a, b, c = axis
        C = self._cos(theta)
        S = self._sin(theta)
        Ccompl = 1 - C
        aa = a * a
        bb = b * b
        cc = c * c
        ab = a * b
        ac = a * c
        bc = b * c
        super().__init__(
            [C + Ccompl * aa, Ccompl * ab + S * c, Ccompl * ac - S * b],
            [Ccompl * ab - S * c, C + Ccompl * bb, Ccompl * bc + S * a],
            [Ccompl * ac + S * b, Ccompl * bc - S * a, C + Ccompl * cc],
        )

if __name__ == '__main__':
    print(Rotation2d.__concrete__)
    print(Rotation2d.__subtypes__)
    print(Rotation2d.__abstract__)
    print(Rotation2d.mro())
    R2 = Rotation2d(0.1)
    R3 = Rotation3d(0.1, (0, 1, 0))
    print(R2)
    print(R3)
