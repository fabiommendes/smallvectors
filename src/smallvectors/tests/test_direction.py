from math import sqrt

import pytest

from smallvectors import Direction2, simeq, Vec2
from .generic_interfaces import LinearFixtures, SequenceInterface
from .mathematical_interfaces import Normed, ElementwiseAddition, \
    ScalarMultiplication
from .vector_interfaces import VectorInterface, VectorInvalidOperations


class TestDirection2D(LinearFixtures,
                      VectorInterface,
                      VectorInvalidOperations,
                      SequenceInterface,
                      ElementwiseAddition,
                      ScalarMultiplication,
                      Normed):
    base_cls = Direction2
    size = LinearFixtures.size

    null = pytest.fixture(lambda self: Vec2(0, 0))
    u = pytest.fixture(lambda self: Direction2(1, 0))
    v = pytest.fixture(lambda self: Direction2(0, 1))
    add_uv = pytest.fixture(lambda self: Vec2(1, 1))
    sub_uv = pytest.fixture(lambda self: Vec2(1, -1))
    double = pytest.fixture(lambda self: Vec2(2, 0))
    half = pytest.fixture(lambda self: Vec2(0.5, 0))

    # Disabled tests
    test_vector_norm_defaults_to_euclidean = None
    test_l1_norm = None

    def test_triangular_identity_2D(self, norm):
        self.assert_triangular_identity(Direction2(1, 2), Direction2(3, 4),
                                        norm)
        self.assert_triangular_identity(Direction2(1, 1), Direction2(1, 1),
                                        norm)
        self.assert_triangular_identity(Direction2(1, 2), Direction2(0, 1),
                                        norm)

    def test_direction_is_always_unitary(self):
        u = Direction2(1, 1)
        assert simeq(u.x, 1 / sqrt(2))
        assert simeq(u.y, 1 / sqrt(2))

    def test_sum_of_two_directions_is_a_vector(self):
        u = Direction2(1, 0)
        v = Direction2(0, 1)
        assert isinstance(u + v, Vec2)
        assert u + v == Vec2(1, 1)

    def test_floordiv(self):
        d = Direction2(1, 1)
        x, y = d // 2
        assert x == y == 0
