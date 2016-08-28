"""
These abstract tests can be imported in modules that implement normed objects.
"""

import pytest


@pytest.fixture
def tol():
    return 1e-5


@pytest.fixture
def norm():
    return None


def assert_triangular_identity(a, b, norm):
    assert (a + b).norm(norm) <= a.norm(norm) + b.norm(norm)


def test_unit_object_has_unity_norm(unitary, tol, norm):
    assert abs(unitary.norm(norm) - 1.0) < tol
    assert abs(unitary.norm_sqr(norm) - 1.0) < tol
    assert unitary.is_unity(norm, tol=tol)


def test_unit_object_is_normalized(unitary, norm, tol):
    assert abs((unitary.normalized(norm) - unitary).norm(norm)) < tol


def test_stretched_object_has_norm_greater_than_one(unitary, norm):
    assert (unitary * 1.1).norm(norm) > 1


def test_shrunk_object_has_norm_smaller_than_one(unitary, norm):
    assert (unitary * 0.9).norm() < 1


def test_triangular_identity(unitary, norm):
    assert_triangular_identity(unitary, unitary, norm)
    assert_triangular_identity(unitary, 2 * unitary, norm)
    assert_triangular_identity(unitary, 0 * unitary, norm)


def test_null_vector_is_null(unitary):
    assert not unitary.is_null()
    assert (unitary * 0).is_null()
