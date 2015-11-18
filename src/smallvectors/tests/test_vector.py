import pytest
from smallvectors import Vec, mVec, VecAny
from generic import op


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
    assert issubclass(Vec, VecAny)
    assert issubclass(mVec, VecAny)


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
def test_norm():
    u = Vec(3, 4)
    assert u.norm() == 5.0
    assert abs(u.normalized() - Vec(3 / 5, 4 / 5)) < 1e-6
    assert u.norm() == abs(u)


def test_clamp():
    u = Vec(3, 4)
    assert u.clampped(1, 10) == u
    assert u.clampped(10) == 2 * u
    assert u.clampped(2, 4) == u.normalized() * 4


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
