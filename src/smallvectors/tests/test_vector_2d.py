from math import pi, sqrt

import pytest

from smallvectors import Vec2, simeq
from .generic_interfaces import LinearFixtures, SequenceInterface
from .mathematical_interfaces import ElementwiseAddition, ScalarMultiplication, \
    Normed
from .vector_interfaces import VectorInterface, VectorInvalidOperations


class TestVector2DInterfaces(LinearFixtures,
                             VectorInterface,
                             VectorInvalidOperations,
                             SequenceInterface,
                             ElementwiseAddition,
                             ScalarMultiplication,
                             Normed):
    base_cls = Vec2

    def test_vector_repr(self):
        assert repr(Vec2(1, 2)) == 'Vec2(1, 2)'
        assert repr(Vec2(0.5, 0.5)) == 'Vec2(0.5, 0.5)'

    def test_polar_coordinates(self):
        assert Vec2.from_polar(1, 0) == Vec2(1, 0)

    def test_rotations(self):
        v = Vec2(1, 0)
        assert simeq(v.rotated_by(pi / 2), Vec2(0, 1))
        assert simeq(v.rotated_axis(pi / 2, Vec2(1, 0)), v)

    def test_cross(self):
        assert Vec2(1, 0).cross(Vec2(0, 1)) == 1

    def test_polar(self):
        r, t = Vec2(1, 1).polar()
        assert simeq(r, sqrt(2))
        assert simeq(t, pi / 4)

    def test_perp(self):
        assert Vec2(1, 0).perpendicular() == Vec2(0, 1)
        assert Vec2(1, 0).perpendicular(ccw=False) == Vec2(0, -1)

    def test_rotated_is_new(self, u):
        assert u.rotated_by(0.0) is not u
        assert u.rotated_by(0.0) == u

    def test_rotated_keeps_norm(self, u):
        for t in range(5):
            Z1 = u.norm()
            Z2 = u.rotated_by(6.28 * t / 5).norm()
            assert simeq(Z1, Z2)

    def test_triangular_identity_2D(self, norm):
        self.assert_triangular_identity(Vec2(1, 2), Vec2(3, 4), norm)
        self.assert_triangular_identity(Vec2(1, 1), Vec2(1, 1), norm)
        self.assert_triangular_identity(Vec2(1, 2), Vec2(0, 0), norm)


class TestVec2Examples:
    def test_class_parameters(self):
        vec2 = Vec2
        assert vec2.shape == (2,)
        assert vec2.size == 2
        assert vec2.dtype == float
        assert vec2.__name__ == 'Vec2'

    def test_correct_type_promotion_on_vec_creation(self):
        assert isinstance(Vec2(1.0, 2.0), Vec2)
        assert isinstance(Vec2(1, 2.0), Vec2)
        assert isinstance(Vec2(1.0, 2), Vec2)

    def test_vec_equality(self):
        assert Vec2(1, 2) == Vec2(1, 2)
        assert Vec2(1, 2) == Vec2(1.0, 2.0)

    def test_vec_equality_with_tuples_and_lists(self):
        assert Vec2(1, 2) == [1, 2]
        assert Vec2(1, 2) == (1, 2)
        assert Vec2(1, 2) == [1.0, 2.0]
        assert Vec2(1, 2) == (1.0, 2.0)

    def test_reverse_vec_equality_with_tuples_and_lists(self):
        assert [1, 2] == Vec2(1, 2)
        assert (1, 2) == Vec2(1, 2)
        assert [1.0, 2.0] == Vec2(1, 2)
        assert (1.0, 2.0) == Vec2(1, 2)

    def test_vec_type_promotion_on_arithmetic_operations(self):
        u = Vec2(1, 2)
        v = Vec2(0.0, 0.0)
        assert isinstance(u + v, Vec2)
        assert isinstance(u - v, Vec2)
        assert isinstance(u * 1.0, Vec2)
        assert isinstance(u / 1.0, Vec2)


class TestRegressions:
    def test_clamp(sef):
        u = Vec2(3, 4)
        assert u.clamped(1, 10) == u
        assert u.clamped(10) == 2 * u
        assert u.clamped(2, 4) == u.normalized() * 4

    def test_has_flat(self):
        u = Vec2(1, 2)
        v = Vec2(1, 2.0)
        assert u.flat[0] == 1
        assert u.flat[:] == [1, 2]
        assert v.flat[-1] == 2.0
        assert v.flat[:] == [1.0, 2.0]
