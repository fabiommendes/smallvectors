from math import pi, sqrt

from smallvectors import Vec3, simeq
from .generic_interfaces import LinearFixtures, SequenceInterface
from .mathematical_interfaces import ElementwiseAddition, ScalarMultiplication, \
    Normed
from .vector_interfaces import VectorInterface, VectorInvalidOperations


class TestVector3DInterfaces(LinearFixtures,
                             VectorInterface,
                             VectorInvalidOperations,
                             SequenceInterface,
                             ElementwiseAddition,
                             ScalarMultiplication,
                             Normed):
    base_cls = Vec3

    def test_vector_repr(self):
        assert repr(Vec3(1, 2, 4)) == 'Vec3(1, 2, 4)'
        assert repr(Vec3(0.5, 0.5, 0.5)) == 'Vec3(0.5, 0.5, 0.5)'

    def test_cross(self):
        assert Vec3(1, 0, 0).cross(Vec3(0, 1, 0)) == Vec3(0, 0, 1)

    def test_spherical_coordinates(self):
        r, phi, theta = Vec3(1, 1, 0).spherical()
        assert simeq(r, sqrt(2))
        assert simeq(phi, pi / 2)
        assert simeq(theta, pi / 4)

    def test_triangular_identity_3D(self, norm):
        self.assert_triangular_identity(Vec3(1, 2, 3), Vec3(3, 4, 5), norm)
        self.assert_triangular_identity(Vec3(1, 1, 1), Vec3(1, 1, 1), norm)
        self.assert_triangular_identity(Vec3(1, 2, 3), Vec3(0, 0, 0), norm)


class TestVec3Examples:
    def test_class_parameters(self):
        vec2 = Vec3
        assert vec2.shape == (3,)
        assert vec2.size == 3
        assert vec2.dtype == float
        assert vec2.__name__ == 'Vec3'

    def test_correct_type_promotion_on_vec_creation(self):
        assert isinstance(Vec3(1.0, 2.0, 3.0), Vec3)
        assert isinstance(Vec3(1, 2.0, 3.0), Vec3)
        assert isinstance(Vec3(1.0, 2, 3), Vec3)

    def test_vec_equality(self):
        assert Vec3(1, 2, 3) == Vec3(1, 2, 3)
        assert Vec3(1, 2, 3) == Vec3(1.0, 2.0, 3.0)

    def test_vec_equality_with_tuples_and_lists(self):
        assert Vec3(1, 2, 3) == [1, 2, 3.0]
        assert Vec3(1, 2, 3) == (1, 2, 3.0)
        assert Vec3(1, 2, 3) == [1.0, 2.0, 3.0]
        assert Vec3(1, 2, 3) == (1.0, 2.0, 3.0)

    def test_reverse_vec_equality_with_tuples_and_lists(self):
        assert [1, 2, 3] == Vec3(1, 2, 3)
        assert (1, 2, 3) == Vec3(1, 2, 3)
        assert [1.0, 2.0, 3.0] == Vec3(1, 2, 3)
        assert (1.0, 2.0, 3.0) == Vec3(1, 2, 3)

    def test_vec_type_promotion_on_arithmetic_operations(self):
        u = Vec3(1, 2, 3)
        v = Vec3(0.0, 0.0, 0.0)
        assert isinstance(u + v, Vec3)
        assert isinstance(u - v, Vec3)
        assert isinstance(u * 1.0, Vec3)
        assert isinstance(u / 1.0, Vec3)
