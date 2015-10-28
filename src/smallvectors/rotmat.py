from .matrix import MatAny, Mat, mMat  

class RotMat2DAny(MatAny[2, 2]):

    '''Cria uma matriz de rotação que realiza a rotação pelo ângulo theta
    especificado'''

    __slots__ = ['_theta', '_transposed']

    def __init__(self, theta):
        self._theta = float(theta)
        self._transposed = None

        C = m.cos(theta)
        S = m.sin(theta)
        M = [[C, -S], [S, C]]
        super(RotMat2, self).__init__(M)

    def rotate(self, theta):
        return RotMat2(self.theta + theta)

    def transpose(self):
        if self._transposed is None:
            self._transposed = super(RotMat2, self).transpose()
        return self._transposed

    def inv(self):
        return self.transpose()

    @property
    def theta(self):
        return self.theta



class RotMat2(Mat2):

    '''Cria uma matriz de rotação que realiza a rotação pelo ângulo theta
    especificado'''

    __slots__ = ['_theta', '_transposed']

    def __init__(self, theta):
        self._theta = float(theta)
        self._transposed = None

        C = m.cos(theta)
        S = m.sin(theta)
        M = [[C, -S], [S, C]]
        super(RotMat2, self).__init__(M)

    def rotate(self, theta):
        return RotMat2(self.theta + theta)

    def transpose(self):
        if self._transposed is None:
            self._transposed = super(RotMat2, self).transpose()
        return self._transposed

    def inv(self):
        return self.transpose()

    @property
    def theta(self):
        return self.theta


class RotMat3:

    '''
        Cria uma matriz de rotação que realiza a rotação pelo ângulo theta
        especificado
    '''
    __slots__ = ['_theta', '_transposed']

    def __init__(self, theta, axis):
        self._theta = float(theta)
        self._transposed = None

        C = m.cos(theta)
        S = m.sin(theta)

        if isinstance(axis, str) and axis == 'x':
            M = [[1, 0, 0], [0, C, - S], [0, S, C]]
        elif isinstance(axis, str) and axis == 'y':
            M = [[C, 0, S], [0, 1, 0], [- S, 0, C]]
        elif isinstance(axis, str) and axis == 'z':
            M = [[C, - S, 0], [S, C, 0], [0, 0, 1]]
        elif isinstance(axis, Vec3):
            M = self._rotate_by_vector_axis(theta, axis)
        else:
            raise InvalidAxisError("Eixo '" + axis + "' invalido.")

        super(RotMat3, self).__init__(M)

    def rotate(self, theta):
        return RotMat3(self._theta + theta)

    def transpose(self):
        if self._transposed is None:
            self._transposed = super(RotMat3, self).transpose()
        return self._transposed

    def inv(self):
        return self.transpose()

    @property
    def theta(self):
        return self._theta

    def _rotate_by_vector_axis(self, theta, vector):
        a, b, c = vector.as_tuple()
        C = m.cos(theta)
        S = m.sin(theta)

        line1 = [(C + (1 - C) * (a ** 2)),
                 (((1 - C) * a * b) + (S * c)),
                 (((1 - C) * a * c) - (S * b))]
        line2 = [(((1 - C) * b * a) - (S * c)),
                 (C + ((1 - C) * (b ** 2))),
                 (((1 - C) * b * c) + (S * a))]
        line3 = [(((1 - C) * c * a) + (S * b)),
                 (((1 - C) * c * b) - (S * a)),
                 (C + ((1 - C) * (c ** 2)))]

        M = [line1, line2, line3]
        return M



