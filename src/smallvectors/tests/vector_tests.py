# -*- coding: utf8 -*-

'''
testing vectors
'''

from math import pi
from nose.tools import raises, assert_almost_equal, assert_almost_equals
from unittest import TestCase
from smallvectors import Vec


class lazy(object):

    '''Implementa uma propriedade "preguiçosa": ela é calculada apenas durante o
    primeiro uso e não durante a inicialização do objeto.'''

    def __init__(self, force):
        self.force = force

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        value = self.force(obj)
        setattr(obj, self.force.__name__, value)
        return value


class Immutable(object):
    vector = None

    @lazy
    def u(self):
        return self.vector(1, 2)

    @lazy
    def v(self):
        return self.vector(2, 1)

    # Operações matemáticas inválidas #########################################

    # Propriedades de vetores e operações geométricas #########################
    def test_vector_norm(self):
        v = self.vector(3, 4)
        assert_almost_equal(v.norm(), 5)
        assert_almost_equal(v.norm_sqr(), 25)

    def test_rotated(self):
        v = self.vector(1, 2)
        v_rot = v.rotate(pi / 2)
        v_res = self.vector(-2, 1)
        assert_almost_equal((v_rot - v_res).norm(), 0)
        assert_almost_equal((v_rot.rotate(pi / 2) + v).norm(), 0)
        assert_almost_equal(v.norm(), v.rotate(pi / 2).norm())

    def test_normalized(self):
        v = self.vector(1, 2)
        n = v.normalized()

        assert_almost_equal(v.normalized().norm(), 1)
        assert_almost_equals(n.x * v.x + n.y * v.y, v.norm())

    # Interface Python ########################################################
    def test_as_tuple(self):
        v = self.vector(1, 2)
        t = v.as_tuple()
        assert isinstance(t, tuple)
        assert t == (1, 2)

    def test_getitem(self):
        v = self.vector(1, 2)
        assert v[0] == 1, v[1] == 2

    @raises(IndexError)
    def test_overflow(self):
        v = self.vector(1, 2)
        v[2]

    def test_iter(self):
        v = self.vector(1, 2)
        assert list(v) == [1, 2]

    def test_len(self):
        v = self.vector(1, 2)
        assert len(v) == 2


class Mutable(Immutable):
    # Modificação de coordenadas ##############################################

    def test_set_coords(self):
        v = self.vector(1, 2)
        v.x = 2
        assert v == (2, 2)

    def test_setitem(self):
        v = self.vector(1, 2)
        v[0] = 2
        assert v == (2, 2)

    def test_update(self):
        v = self.vector(1, 2)
        v.update(2, 1)
        assert v == (2, 1)
        v.update((1, 2))
        assert v == (1, 2)

    # Operações matemáticas inplace ###########################################
    def test_iadd(self):
        v = self.vector(1, 2)
        v += (1, 2)
        assert v == (2, 4)

    def test_imul(self):
        v = self.vector(1, 2)
        v *= 2
        assert v == (2, 4)

    def test_idiv(self):
        v = self.vector(1, 2)
        v /= 0.5
        assert v == (2, 4)

###############################################################################
#                                TestCases
###############################################################################


class FloatVecTest(Immutable, TestCase):
    vector = Vec[2, float]


# class CVectorMTest(Mutable, TestCase):
#    from FGAme.mathutils.cvector import mVec2 as vector


# class PyVectorTest(Immutable, TestCase):
#    from FGAme.mathutils.vector import Vec2 as vector


# class PyVectorMTest(Mutable, TestCase):
#    from FGAme.mathutils.vector import mVec2 as vector


u = Vec(1, 2)
print(u * u)

if __name__ == '__main__':
    import nose
    nose.runmodule('__main__')
