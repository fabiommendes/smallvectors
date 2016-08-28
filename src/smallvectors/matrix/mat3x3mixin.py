from generic import promote_type

from smallvectors import Vec
from smallvectors.matrix.square import SquareMixin


class Mat3x3Mixin(SquareMixin):
    __slots__ = ('flat',)

    def det(self):
        a, b, c, d, e, f, g, h, i = self.flat
        d1 = + (a * e * i)
        d2 = + (b * f * g)
        d3 = + (c * d * h)
        d4 = - (c * e * g)
        d5 = - (a * f * h)
        d6 = - (b * d * i)
        return d1 + d2 + d3 + d4 + d5 + d6

    def trace(self):
        return self.flat[0] + self.flat[4] + self.flat[8]

    def diag(self):
        return Vec[3, self.dtype](self.flat[0], self.flat[4], self.flat[8])

    def transpose(self):
        a, b, c, d, e, f, g, h, i = self.flat
        return self.fromflat([a, d, g,
                              b, e, h,
                              c, f, i], copy=False)

    def inv(self):
        Z = 1 / self.det()
        a, b, c, d, e, f, g, h, i = self.flat
        data = [
            (e * i - f * h) * Z, (c * h - b * i) * Z, (b * f - c * e) * Z,
            (f * g - d * i) * Z, (a * i - c * g) * Z, (c * d - a * f) * Z,
            (d * h - e * g) * Z, (b * g - a * h) * Z, (a * e - b * d) * Z,
        ]
        return self.fromflat(data, copy=False)

    # TODO: use these faster versions
    def __mul_matrix(self, other):
        outtype = promote_type(type(self), type(other))
        a, b, c, d, e, f, g, h, i = self.flat
        j, k, l, m, n, o, p, q, r = other.flat
        data = [
            a * j + b * m + c * p, a * k + b * n + c * q, a * l + b * o + c * r,
            d * j + e * m + f * p, d * k + e * n + f * q, d * l + e * o + f * r,
            g * j + h * m + i * p, g * k + h * n + i * q, g * l + h * o + i * r,
        ]
        return outtype.fromflat(data, copy=False)

    def __mul_vector(self, other):
        a, b, c, d, e, f, g, h, i = self.flat
        x, y, z = other

        return Vec(a * x + d * y + g * z,
                   b * x + e * y + h * z,
                   c * x + f * y + i * z)


class mMat3x3:
    def itranspose(self):
        """
        Transpose matrix *INPLACE**:
        """

        data = self.flat
        a, b, c = data[1], data[2], data[5]
        data[1], data[2] = data[3], data[6]
        data[3], data[5] = a, data[7]
        data[6], data[7] = b, c


