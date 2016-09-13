from math import pi, sqrt
from random import random as rand

import pytest
from generic import op

from smallvectors import Vec, mVec, simeq
from smallvectors.tests import abstract as base, arithmetic
from smallvectors.vector.linear import LinearAny


class VectorBase(base.TestNormedObject,
                 base.TestMutability,
                 arithmetic.TestPairwiseAddition):
    base_cls = Vec

    @property
    def base_args__add_ab(self):
        return (x + y for (x, y) in zip(self.base_args, self.base_args__other))

    @property
    def base_args__sub_ab(self):
        return (x - y for (x, y) in zip(self.base_args, self.base_args__other))

    @property
    def base_args__zero(self):
        return (0 for x in self.args())

    @pytest.fixture
    def unitary(self, args, kwargs):
        return Vec(*args, **kwargs)

    @pytest.fixture
    def null(self, args):
        return Vec(*(0 for x in args))

    @pytest.fixture
    def u(self, cls, args, kwargs):
        return cls(*(rand() for x in args), **kwargs)

    @pytest.fixture
    def v(self, cls, args, kwargs):
        return cls(*(rand() for x in args), **kwargs)

    def test_mutable_cls_is_mVec(self, mutable_cls):
        assert mutable_cls is mVec

    # Tests conversions
    def test_conversion_complex(self, unitary):
        conv = unitary.convert(complex)
        assert type(conv) is not type(unitary)

    # Arithmetic tests
    def test_invalid_add_scalar(self, a):
        with pytest.raises(TypeError):
            y = a + 1

    def test_invalid_sub_scalar(self, a):
        with pytest.raises(TypeError):
            y = a - 1

    def test_invalid_mul_tuple(self, a):
        with pytest.raises(TypeError):
            y = a * (1, 2)

    def test_invalid_mul_vec(self, a):
        with pytest.raises(TypeError):
            y = a * a

    def test_invalid_div_tuple(self, a):
        with pytest.raises(TypeError):
            y = a / (1, 2)
        with pytest.raises(TypeError):
            y = (1, 2) / a

    def test_invalid_div_vec(self, a):
        with pytest.raises(TypeError):
            y = a / a

    def test_invalid_div_scalar(self, a):
        with pytest.raises(TypeError):
            y = 1 / a

    def test_vec_almost_equal(self, a, b):
        b = a + b / 1e9
        w = a + 0 * b
        assert a.almost_equal(b)
        assert b.almost_equal(a)
        assert a.almost_equal(w)
        assert w.almost_equal(a)

    # Test API
    def test_clamp_to_value(self, unitary):
        assert simeq(unitary.clamp(2), 2 * unitary)
        assert simeq(unitary.clamp(0.5), 0.5 * unitary)

    def test_clamp_interval(self, unitary):
        assert unitary.clamp(0.5, 2) == unitary

    def test_clamp_missing_interval(self, unitary):
        assert simeq(unitary.clamp(2, 3), 2 * unitary)
        assert simeq(unitary.clamp(0.1, 0.5), 0.5 * unitary)

    def test_lerp(self, u, v):
        assert simeq(u.lerp(v), v.lerp(u))
        assert simeq(u.middle(v), u.lerp(v))
        assert simeq(u.lerp(v, 0), u)
        assert simeq(u.lerp(v, 1), v)

    def test_middle(self, unitary, null):
        assert simeq(unitary.middle(null), null.middle(unitary))
        assert simeq(unitary.middle(null), unitary / 2)

    def test_distance(self, unitary, null):
        assert simeq(unitary.distance(unitary), 0)
        assert simeq(unitary.distance(null), 1)
        assert simeq(unitary.distance(-unitary), 2)

    def test_angle(self, unitary):
        assert simeq(unitary.angle(unitary), 0)
        assert simeq(unitary.angle(-unitary), pi)

    def test_vector_norm_defaults_to_euclidean(self, cls, args):
        vec = cls(*(1 for _ in args))
        assert simeq(vec.norm(), sqrt(len(args)))
        assert simeq(abs(vec), sqrt(len(args)))


class TestVector2D(VectorBase):
    base_args = (0.6, 0.8)
    base_args__other = (1, 2)

    def test_polar_coordinates(self):
        assert Vec[2, float].from_polar(1, 0) == Vec(1, 0)

    def test_rotations(self):
        v = Vec(1, 0)
        assert simeq(v.rotate(pi / 2), Vec(0, 1))
        assert simeq(v.rotate_at(pi / 2, Vec(1, 0)), v)

    def test_cross(self):
        assert Vec(1, 0).cross(Vec(0, 1)) == 1

    def test_polar(self):
        r, t = Vec(1, 1).polar()
        assert simeq(r, sqrt(2))
        assert simeq(t, pi / 4)

    def test_perp(self):
        assert Vec(1, 0).perp() == Vec(0, 1)
        assert Vec(1, 0).perp(cw=True) == Vec(0, -1)

    def test_triangular_identity_2D(self):
        self.assert_triangular_identity(Vec(1, 2), Vec(3, 4), None)
        self.assert_triangular_identity(Vec(1, 1), Vec(1, 1), None)
        self.assert_triangular_identity(Vec(1, 2), Vec(0, 0), None)

    def test_rotated_is_new(self, a):
        assert a.rotate(0.0) is not a
        assert a.rotate(0.0) == a

    def test_rotated_keeps_norm(self, obj):
        for t in range(5):
            Z1 = obj.norm()
            Z2 = obj.rotate(6.28 * t / 5).norm()
            assert abs(Z1 - Z2) < 1e-6, (Z1, Z2)


class TestVector3D(VectorBase):
    base_args = (1 / 3, 2 / 3, 2 / 3)
    base_args__other = (1, 2, 3)

    def test_triangular_identity_3D(self):
        self.assert_triangular_identity(Vec(1, 2, 3), Vec(4, 5, 6), None)
        self.assert_triangular_identity(Vec(1, 2, 3), Vec(1, 2, 3), None)
        self.assert_triangular_identity(Vec(1, 2, 3), Vec(0, 0, 0), None)


class TestVector4D(VectorBase):
    base_args = (0.5, 0.5, 0.5, 0.5)
    base_args__other = (1, 2, 3, 4)


class TestVector5D(VectorBase):
    base_args = (0.4, 0.4, 0.4, 0.4, 0.6)
    base_args__other = (1, 2, 3, 4, 5)


# Test Vec type properties
def test_dispatch():
    assert (Vec[2, float], Vec[2, int]) in op.add
    assert (Vec[2, float], int) in op.mul
    assert (int, Vec[2, float]) in op.mul


def test_subclass():
    assert issubclass(Vec, LinearAny)
    assert issubclass(mVec, LinearAny)


def test_unique_subclass():
    assert Vec[2, float] is Vec[2, float]
    assert Vec[2, int] is Vec[2, int]


def test_class_parameters():
    vec2 = Vec[2, float]
    assert vec2.shape == (2,)
    assert vec2.size == 2
    assert vec2.dtype == float
    assert vec2.__parameters__ == (2, float)
    assert vec2.__name__ == 'Vec[2, float]'


# Instance tests
def test_correct_type_promotion_on_vec_creation():
    assert isinstance(Vec(1, 2), Vec[2, int])
    assert isinstance(Vec(1.0, 2.0), Vec[2, float])
    assert isinstance(Vec(1, 2.0), Vec[2, float])
    assert isinstance(Vec(1.0, 2), Vec[2, float])


def test_vec_equality():
    assert Vec(1, 2) == Vec(1, 2)
    assert Vec(1, 2) == Vec(1.0, 2.0)


def test_vec_equality_with_tuples_and_lists():
    assert Vec(1, 2) == [1, 2]
    assert Vec(1, 2) == (1, 2)
    assert Vec(1, 2) == [1.0, 2.0]
    assert Vec(1, 2) == (1.0, 2.0)


def test_reverse_vec_equality_with_tuples_and_lists():
    assert [1, 2] == Vec(1, 2)
    assert (1, 2) == Vec(1, 2)
    assert [1.0, 2.0] == Vec(1, 2)
    assert (1.0, 2.0) == Vec(1, 2)


def test_vec_type_promotion_on_arithmetic_operations():
    u = Vec(1, 2)
    v = Vec(0.0, 0.0)
    assert isinstance(u + v, Vec[2, float])
    assert isinstance(u - v, Vec[2, float])
    assert isinstance(u * 1.0, Vec[2, float])
    assert isinstance(u / 1.0, Vec[2, float])


# Regression testing
def test_clamp():
    u = Vec(3, 4)
    assert u.clamp(1, 10) == u
    assert u.clamp(10) == 2 * u
    assert u.clamp(2, 4) == u.normalize() * 4


def test_mixed_vec_types():
    u = Vec[2, float](1, 2)
    v = Vec[2, int](1, 2)
    assert u + v == v + u


def test_has_flat():
    u = Vec(1, 2)
    v = Vec(1, 2.0)
    w = Vec(1, 2, 3, 4, 5, 6)
    assert u.flat[0] == 1
    assert u.flat[:] == [1, 2]
    assert v.flat[-1] == 2.0
    assert v.flat[:] == [1.0, 2.0]
    assert w.flat[-1] == 6
    assert w.flat[::2] == [1, 3, 5]
