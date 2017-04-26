import pytest

from smallvectors import Point2, Vec2
from .generic_interfaces import LinearFixtures, SequenceInterface
from .mathematical_interfaces import ElementwiseAddition, ScalarMultiplication, \
    Normed
from .vector_interfaces import VectorInterface, VectorInvalidOperations


class TestPoint2D(LinearFixtures,
                  VectorInvalidOperations,
                  SequenceInterface):
    base_cls = Point2
    add_uv = ElementwiseAddition.add_uv
    null = pytest.fixture(lambda self: Vec2(0, 0))

    # Disabled tests
    test_vec_almost_equal = None

    def test_addition(self, u, v, add_uv):
        assert u + Vec2(*v) == add_uv
        assert Vec2(*v) + u == add_uv
