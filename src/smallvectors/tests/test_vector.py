import pytest

from generic import op
from smallvectors import Vec, mVec
from smallvectors.vector.linear import LinearAny


#
# Vector norm
#
@pytest.fixture
def unitary():
    return Vec(0.6, 0.8)

from smallvectors.tests.normtests import (
    norm, tol,
    test_unit_object_has_unity_norm,
    test_triangular_identity,
    test_unit_object_is_normalized,
    test_shrunk_object_has_norm_smaller_than_one,
    test_stretched_object_has_norm_greater_than_one,
    test_null_vector_is_null,
    assert_triangular_identity,
)


def test_vector_norm_defaults_to_euclidean(tol):
    u = Vec(3, 4)
    assert u.norm() == 5.0
    assert abs(u.normalize() - Vec(3 / 5, 4 / 5)) < tol
    assert u.norm() == abs(u)


def test_triangular_identity_2D():
    assert_triangular_identity(Vec(1, 2), Vec(3, 4), None)
    assert_triangular_identity(Vec(1, 1), Vec(1, 1), None)
    assert_triangular_identity(Vec(1, 2), Vec(0, 0), None)


def test_triangular_identity_3D():
    assert_triangular_identity(Vec(1, 2, 3), Vec(4, 5, 6), None)
    assert_triangular_identity(Vec(1, 2, 3), Vec(1, 2, 3), None)
    assert_triangular_identity(Vec(1, 2, 3), Vec(0, 0, 0), None)


#
# Test Vec type properties
#
def test_class_is_type():
    u = Vec(1, 2)
    assert u.__class__ == type(u)


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
    assert vec2.__name__ ==  'Vec[2, float]'


#
# Instance tests
#
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


#
# Regression testing
#
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

if __name__ == '__main__':
    pytest.main('test_vector.py -q')
