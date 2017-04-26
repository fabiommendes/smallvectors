import operator as op

import pytest

from smallvectors.utils import simeq


class ElementwiseAddition:
    """
    For objects that implement a elementwise addition and subtraction.
    """

    @pytest.fixture
    def equiv(self):
        return [tuple, list]

    @pytest.fixture
    def add_uv(self, u, v):
        return self.base_cls(*get_args(u, v, op.add))

    @pytest.fixture
    def sub_uv(self, u, v):
        return self.base_cls(*get_args(u, v, op.sub))

    def test_addition(self, u, v, add_uv):
        assert u + v == add_uv

    def test_addition_alts(self, u, v, add_uv, equiv):
        for transform in equiv:
            assert u + transform(v) == add_uv
            assert transform(u) + v == add_uv

    def test_subtraction(self, u, v, sub_uv):
        assert u - v == sub_uv

    def test_subtraction_alts(self, u, v, sub_uv, equiv):
        for transform in equiv:
            assert u - transform(v) == sub_uv
            assert transform(u) - v == sub_uv

    def test_addition_is_commutative(self, u, v):
        assert u + v == v + u

    def test_sub_is_not_commutative(self, u, v):
        assert u - v != v - u

    def test_null_element_does_not_contribute_to_addition(self, u, v, null):
        assert u + null == u
        assert null + u == u
        assert v + null == v
        assert null + v == v
        assert null + null == null

    def test_null_element_does_not_contribute_to_subtraction(self, u, v, null):
        assert u - null == u
        assert v - null == v
        assert null - null == null

    def test_cannot_add_scalar(self, u):
        with pytest.raises((ValueError, TypeError)):
            u + 1
        with pytest.raises((ValueError, TypeError)):
            u - 1
        with pytest.raises((ValueError, TypeError)):
            1 + u
        with pytest.raises((ValueError, TypeError)):
            1 - u

    def test_unary_addition_keeps_vector_the_same(self, u):
        assert +u == u

    def test_unary_subtraction_invariants(self, u):
        assert -u != u
        assert -(-u) == u


class ScalarMultiplication:
    """
    For objects that implement scalar multiplication/division.
    """

    @pytest.fixture
    def double(self, u):
        return self.base_cls(*(ui * 2 for ui in u))

    @pytest.fixture
    def half(self, u, v):
        return self.base_cls(*(ui / 2 for ui in u))

    def test_scalar_multiplication_is_commutative(self, u, double):
        assert u * 2 == double
        assert u * 2.0 == double
        assert 2 * u == double
        assert 2.0 * u == double

    def test_multiplication_by_one(self, u, v):
        assert 1 * u == 1.0 * u == u * 1 == u * 1.0 == u
        assert 1 * v == 1.0 * v == v * 1 == v * 1.0 == v

    def test_multiplication_by_zero(self, u, v, null):
        assert 0 * u == u * 0 == 0.0 * u == u * 0.0 == null
        assert 0 * v == v * 0 == 0.0 * v == v * 0.0 == null

    def test_scalar_division(self, u, half):
        assert u / 2 == u / 2.0 == u * 0.5 == half

    def test_scalar_division_by_one(self, u):
        assert u / 1 == u / 1.0 == u

    def test_rhs_division_fails(self, u):
        with pytest.raises((ValueError, TypeError)):
            x = 1 / u
        with pytest.raises((ValueError, TypeError)):
            x = 1 // u

    def test_vec_vec_division_fails(self, u):
        with pytest.raises((ValueError, TypeError)):
            x = u / u
        with pytest.raises((ValueError, TypeError)):
            x = u // u


def get_args(u, v, op):
    return tuple(op(ui, vi) for (ui, vi) in zip(u, v))


class Normed:
    """
    Abstract tests for normed objects.
    """

    @pytest.fixture
    def norm(self):
        return 'euclidean'

    @pytest.fixture
    def tol(self):
        return 1e-6

    def assert_triangular_identity(self, u, v, norm):
        norm_sum = (u + v).norm(norm)
        sum_norm = u.norm(norm) + v.norm(norm)
        assert norm_sum <= sum_norm + 1e-6

    def test_unit_object_has_unity_norm(self, unity, tol, norm):
        assert abs(unity.norm(norm) - 1.0) < tol
        assert abs(unity.norm_sqr(norm) - 1.0) < tol
        assert unity.is_unity(norm, tol=tol)

    def test_unity_object_under_high_tolerance(self, unity):
        assert (0.5 * unity).is_unity(tol=1.0)

    def test_doubled_object_is_not_normalized(self, unity, tol):
        assert abs((2 * unity).norm() - 2) < tol

    def test_unit_object_is_normalized(self, unity, tol, norm):
        assert abs((unity.normalized(norm) - unity).norm(norm)) < tol

    def test_stretched_object_has_norm_greater_than_one(self, unity, norm):
        assert (unity * 1.1).norm(norm) > 1

    def test_shrunk_object_has_norm_smaller_than_one(self, unity, norm):
        assert (unity * 0.9).norm(norm) < 1

    def test_triangular_identity_unity_vector(self, unity, norm):
        self.assert_triangular_identity(unity, unity, norm)
        self.assert_triangular_identity(unity, 2 * unity, norm)
        self.assert_triangular_identity(unity, 0 * unity, norm)

    def test_conversion_to_normalized(self, unity):
        assert simeq(unity, unity.normalized())

    def test_null_vector_is_null(self, unity):
        assert not unity.is_null()
        assert (unity * 0).is_null()

    def test_null_vector_with_tolerance(self, unity):
        assert unity.is_null(tol=1)
