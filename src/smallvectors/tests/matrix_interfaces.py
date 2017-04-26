import copy

import pytest

from smallvectors import Vec2, Vec


class MatrixInterface:
    base_cls = None
    shape = pytest.fixture(lambda self: self.base_cls.shape)
    M1 = pytest.fixture(lambda self: NotImplemented)
    M2 = pytest.fixture(lambda self: NotImplemented)
    u = pytest.fixture(lambda self, M1: M1)
    v = pytest.fixture(lambda self, M2: M2)

    @pytest.fixture
    def null(self, shape):
        m, n = shape
        return self.base_cls(*([0] * n for _ in range(m)))

    def test_matrix_equality(self, M1):
        assert M1 == copy.copy(M1)

    def test_create_from_cols(self, M1):
        cols = list(M1.T)
        from_cols = self.base_cls.from_cols(cols)
        assert M1 == from_cols

    def test_create_from_rows(self, M1):
        rows = list(M1)
        assert M1 == self.base_cls.from_rows(rows)

    def test_create_from_empty_dict(self, null, shape):
        m, n = shape
        mat = self.base_cls.from_dict({})
        assert mat == null

    @pytest.mark.skip
    def test_create_from_dict(self, null, shape):
        m, n = shape
        mat = self.base_cls[m, n].from_dict({(0, 0): 1})
        assert mat != null
        assert mat[0, 0] == 1

    # Conversions
    def test_conversion_to_dict(self, M1):
        D = M1.as_dict()
        assert isinstance(D, dict)
        assert 0 not in D.values()
        assert D
        assert type(M1).from_dict(D) == M1

    def test_conversion_to_items(self, M1):
        D = M1.as_dict()
        items = {k: v for k, v in M1.items() if v}
        assert D == items

    # Accessors
    def test_col_accessor(self, M1):
        col = M1.col(0)
        assert isinstance(col, Vec)
        assert col == M1.T[0]

    def test_row_accessor(self, M1):
        row = M1.row(0)
        assert isinstance(row, Vec)
        assert row == M1[0]

    # Copy
    def test_copy_changing_terms(self, M1):
        assert M1 == M1.copy()

        cp = M1.copy(A00=42)
        assert cp[0, 0] == 42
        assert M1 == cp.copy(A00=M1[0, 0])

    # Shape transformations
    @pytest.mark.skip
    def test_append_column(self, M1, shape):
        m, n = shape
        col = M1.col(0)
        M2 = M1.append_col(col)
        assert M2.shape == (m, n + 1)
        assert M2[0, n] == col[0]


class SquareMatrixInterface:
    @pytest.fixture
    def N(self, shape):
        return shape[0]

    @pytest.fixture
    def I(self, cls, shape):
        N = shape[0]
        return cls[N, N].from_diag([1] * N)

    def test_shape_is_square(self, obj, shape):
        m, n = obj.shape
        cls = type(obj)
        assert m == n
        assert cls.shape == obj.shape
        if shape is not None:
            assert obj.shape == shape

    def test_shape_is_not_changed_by_transposition(self, M1):
        assert M1.shape == M1.T.shape

    def test_create_from_diag(self, N):
        diag = [1] * N
        M = self.base_cls.from_diag(diag)
        assert M * M == M

    def test_trace_of_identity(self, I, N):
        return I.trace() == N

    def test_identity_diag(self, I, N):
        return list(I.diag()) == [1] * N

    def test_identity_set_null_diag(self, I, N, zero):
        assert I.new_diag([0] * N) == zero

    def test_identity_drop_diag(self, I, zero):
        assert I.new_dropping_diag() == zero

    def test_identity_is_inverse(self, I):
        assert I.inv() == I

    def test_identity_solve(self, I, N):
        b = Vec2(*range(1, N + 1))
        assert I.solve(b) == b
        assert I.solve_jacobi(b) == b
        assert I.solve_gauss(b) == b
        assert I.solve_triangular(b) == b