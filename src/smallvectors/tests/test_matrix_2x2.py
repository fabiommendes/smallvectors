import pytest

from smallvectors import Mat, Mat2
from .matrix_interfaces import MatrixInterface


class TestMat2x2(MatrixInterface):
    base_cls = Mat2
    M1 = pytest.fixture(lambda self: Mat2([1, 2], [3, 4]))
    M2 = pytest.fixture(lambda self: Mat2([2, 1], [4, 3]))
    identity = pytest.fixture(lambda self: Mat2([1, 0], [0, 1]))

    def test_inverse_of_diagonal_matrix(self):
        M = Mat2([2, 0], [0, 4])
        Minv = M.inv()
        assert Minv == Mat2([0.5, 0], [0, 0.25])

    def test_diagonal_inverse_recovers_identity(self):
        M = Mat2([2.0, 0.0], [0.0, 4.0])
        Minv = M.inv()
        assert type(Minv) is type(M)
        assert Minv * M == Mat2([1.0, 0.0], [0.0, 1.0])
        assert M * Minv == Mat2([1.0, 0.0], [0.0, 1.0])

    def test_inverse(self, M1, identity):
        assert M1.inv() * M1 == identity
        assert M1 * M1.inv() == identity

    def test_fromrows_sqr(self):
        M1 = Mat2([1, 2], [3, 4])
        M2 = Mat2.from_rows([[1, 2], [3, 4]])
        assert M1 == M2

    def test_equal_to_list(self):
        M1 = Mat2([1, 2], [3, 4])
        M2 = [[1, 2], [3, 4]]
        assert M1 == M2

    def test_get_item(self):
        M1 = Mat2([1, 2], [3, 4])
        assert M1[0] == [1, 2]
        assert M1[1] == [3, 4]

    def test_flat2x2(self):
        M1 = Mat2([1, 2], [3, 4])
        assert list(M1.flat) == [1, 2, 3, 4]

    def test_flat2x2_redo(self):
        M1 = Mat2([1, 2], [3, 4])
        assert M1, Mat[2, 2].from_flat(M1.flat)

    def test_transpose_sqr(self):
        M1 = Mat2([1, 2], [3, 4])
        M2 = Mat2([1, 3], [2, 4])
        assert M1.T == M2
        assert M1 == M2.T

    def test_widthdiag(self):
        M1 = Mat2([1, 2], [3, 4])
        M2 = Mat2([0, 2], [3, 0])
        assert M1.new_diag([0, 0]) == M2
        assert M1.new_dropping_diag() == M2
