from math import pi, sqrt

import pytest

from smallvectors.utils import simeq


class VectorInterface:
    """
    Generic tests for the vector API
    """

    base_cls = None

    def test_equality_against_tuples_and_lists(self, u):
        assert tuple(u) == u
        assert list(u) == u
        assert u == tuple(u)
        assert u == list(u)

    def test_vec_has_no_dict(self, u):
        with pytest.raises(AttributeError):
            D = u.__dict__

    def test_from_flat_data(self, size):
        args = range(size)
        assert self.base_cls(*args) == self.base_cls.from_flat(args)

    def test_clamp_to_value(self, unity):
        assert simeq(unity.clamped(2), 2 * unity)
        assert simeq(unity.clamped(0.5), 0.5 * unity)

    def test_clamp_interval(self, unity):
        assert unity.clamped(0.5, 2) == unity

    def test_clamp_missing_interval(self, unity):
        assert simeq(unity.clamped(2, 3), 2 * unity)
        assert simeq(unity.clamped(0.1, 0.5), 0.5 * unity)

    def test_lerp(self, u, v):
        assert simeq(u.lerp(v), v.lerp(u))
        assert simeq(u.middle(v), u.lerp(v))
        assert simeq(u.lerp(v, 0), u)
        assert simeq(u.lerp(v, 1), v)

    def test_middle(self, unity, null):
        assert simeq(unity.middle(null), null.middle(unity))
        assert simeq(unity.middle(null), unity / 2)

    def test_distance(self, unity, null):
        assert simeq(unity.distance(unity), 0)
        assert simeq(unity.distance(null), 1)
        assert simeq(unity.distance(-unity), 2)

    def test_angle(self, unity):
        assert simeq(unity.angle(unity), 0)
        assert simeq(unity.angle(-unity), pi)

    def test_vector_norm_defaults_to_euclidean(self, size):
        vec = self.base_cls(*(1 for _ in range(size)))
        assert simeq(vec.norm(), sqrt(size))
        assert simeq(abs(vec), sqrt(size))

    def test_l1_norm(self, size):
        u = self.base_cls(*(1 for _ in range(size)))
        assert abs(u.norm('l1') - size) < 1e-6
        assert abs(u.norm_sqr('l1') - size**2) < 1e-6

    def test_floordiv(self, size):
        u = self.base_cls(*(3 for _ in range(size))) // 2
        assert list(u) == [1] * size, u


class VectorInvalidOperations:
    """
    Tests if invalid vector operations raise the correct errors
    """

    def test_invalid_scalar_operations(self, u):
        with pytest.raises(TypeError):
            y = u + 1
        with pytest.raises(TypeError):
            y = u - 1

    def test_invalid_mul_tuple(self, u):
        with pytest.raises(TypeError):
            y = u * (1, 2)

    def test_invalid_mul_vec(self, u):
        with pytest.raises(TypeError):
            y = u * u

    def test_invalid_div_tuple(self, u):
        with pytest.raises(TypeError):
            y = u / (1, 2)
        with pytest.raises(TypeError):
            y = (1, 2) / u

    def test_invalid_div_vec(self, u):
        with pytest.raises(TypeError):
            y = u / u

    def test_invalid_div_scalar(self, u):
        with pytest.raises(TypeError):
            y = 1 / u

    def test_vec_almost_equal(self, u, v):
        v = u + v / 1e9
        w = u + 0 * v
        assert u.almost_equal(v)
        assert v.almost_equal(u)
        assert u.almost_equal(w)
        assert w.almost_equal(u)