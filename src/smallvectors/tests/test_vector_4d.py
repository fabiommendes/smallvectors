from smallvectors import Vec4
from .generic_interfaces import LinearFixtures, SequenceInterface
from .mathematical_interfaces import ElementwiseAddition, ScalarMultiplication, \
    Normed
from .vector_interfaces import VectorInterface, VectorInvalidOperations


class TestVector4DInterfaces(LinearFixtures,
                             VectorInterface,
                             VectorInvalidOperations,
                             SequenceInterface,
                             ElementwiseAddition,
                             ScalarMultiplication,
                             Normed):
    base_cls = Vec4

    def test_vector_repr(self):
        assert repr(Vec4(1, 2, 4, 5)) == 'Vec4(1, 2, 4, 5)'
        assert repr(Vec4(0.5, 0.5, 0.5, 0.5)) == 'Vec4(0.5, 0.5, 0.5, 0.5)'

    def test_triangular_identity_4D(self, norm):
        self.assert_triangular_identity(Vec4(1, 2, 3, 4), Vec4(3, 4, 5, 6),
                                        norm)
        self.assert_triangular_identity(Vec4(1, 1, 1, 1), Vec4(1, 1, 1, 1),
                                        norm)
        self.assert_triangular_identity(Vec4(1, 2, 3, 4), Vec4(0, 0, 0, 0),
                                        norm)


class TestVec4Examples:
    def test_class_parameters(self):
        assert Vec4.shape == (4,)
        assert Vec4.size == 4
        assert Vec4.dtype == float
        assert Vec4.__name__ == 'Vec4'
