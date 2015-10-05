# -*- coding: utf8 -*-

from collections import MutableSequence
from smallvectors.array import Array
from smallvectors import Vec, asvector

__all__ = ['VecArray']


class VecArray(MutableSequence):
    __slots__ = ['_data']

    def __init__(self, data):
        '''Implementa um array de vetores bidimensionais. As operações
        matemáticas podem ser aplicadas diretamente ao array

        Exemplo
        -------

        Criamos um array inicializando com uma lista de vetores ou uma lista
        de duplas

        >>> a = VecArray([(0, 0), (1, 0), (0, 1)]); a
        VecArray([(0, 0), (1, 0), (0, 1)])

        Sob muitos aspectos, um VecArray funciona como uma lista de vetores

        >>> a[0], a[1]
        (Vec[2, int](0, 0), Vec[2, int](1, 0))

        As operações matemáticas ocorrem termo a termo

        >>> a + (1, 1)
        VecArray([(1, 1), (2, 1), (1, 2)])

        Já as funções de vetores são propagadas para cada elemento e retornam
        um Array numérico ou um VecArray

        >>> a.norm()
        Array([0, 1, 1])
        '''

        self._data = list(asvector(x) for x in data)

    @classmethod
    def _new(cls, data):
        new = cls.__new__(cls)
        new._data = data
        return new

    def as_tuple(self):
        '''Retorna uma lista de tuplas'''
        return [u.as_tuple() for u in self._data]

    def norm(self):
        '''Retorna um Array com o módulo de cada vetor'''

        return Array([u.norm() for u in self._data])

    def norm_sqr(self):
        '''Retorna o módulo do vetor ao quadrado'''

        return Array([u.norm_sqr() for u in self._data])

    def normalized(self):
        '''Retorna um vetor unitário'''

        return VecArray([u.normalized() for u in self._data])

    def rotate(self, theta, axis=None):
        '''Retorna um vetor rotacionado por um ângulo theta'''

        # FIXME: use rotation matrix
        # R = RotMat2(theta)
        if axis is None:
            self._data[:] = [u.rotate(theta) for u in self._data]
        else:
            v = asvector(axis)
            self._data[:] = [v + (u - v).rotate(theta) for u in self._data]

    def move(self, x_or_delta, y=None):
        if y is not None:
            x_or_delta = Vec(x_or_delta, y)
        self._data[:] = [u + x_or_delta for u in self._data]

    # Métodos mágicos #########################################################
    def __len__(self):
        return len(self._data)

    def __repr__(self):
        '''x.__repr__() <==> repr(x)'''

        tname = type(self).__name__
        data = ', '.join([str(tuple(v)) for v in self])
        return '%s([%s])' % (tname, data)

    def __str__(self):
        '''x.__str__() <==> str(x)'''
        return repr(self)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, i):
        '''x.__getitem__(i) <==> x[i]'''

        return self._data[i]

    def __setitem__(self, i, value):
        self._data[i].update(value)

    def __delitem__(self, i):
        del self._data[i]

    def insert(self, idx, value):
        self._data.insert(idx, Vec(value))

    # Operações matemáticas ###################################################
    def __mul__(self, other):
        '''x.__mul__(y) <==> x * y'''

        return self._new(u * other for u in self._data)

    def __rmul__(self, other):
        '''x.__rmul__(y) <==> y * x'''

        return self._new(other * u for u in self._data)

    def __div__(self, other):
        '''x.__div__(y) <==> x / y'''

        return self._new(u / other for u in self._data)

    __truediv__ = __div__  # Python 3

    def __add__(self, other):
        '''x.__add__(y) <==> x + y'''

        other = asvector(other)
        return self._new([u + other for u in self._data])

    def __radd__(self, other):
        '''x.__radd__(y) <==> y + x'''

        return self + other

    def __sub__(self, other):
        '''x.__sub__(y) <==> x - y'''

        other = asvector(other)
        return self._new(u - other for u in self._data)

    def __rsub__(self, other):
        '''x.__rsub__(y) <==> y - x'''

        other = asvector(other)
        return self._new(other - u for u in self._data)

    def __neg__(self):
        '''x.__neg() <==> -x'''

        return self._new(-u for u in self._data)

    def __nonzero__(self):
        return len(self) != 0

    def __eq__(self, other):
        return all(u == other for u in self._data)

    @property
    def x(self):
        return Array([u.x for u in self._data])

    @property
    def y(self):
        return Array([u.y for u in self._data])


if __name__ == '__main__':
    import doctest
    doctest.testmod()
