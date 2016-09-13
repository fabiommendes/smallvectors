import pytest

from smallvectors import Mat, mMat, Rotation2d, Vec
from smallvectors.tests import abstract as base
from smallvectors.tests import arithmetic


class MatrixBase(base.TestMutability,
                 arithmetic.TestPairwiseAddition,
                 arithmetic.TestPairwiseMultiplication,
                 arithmetic.TestScalarMultiplication):
    base_cls = Mat
    base_shape = None

    @pytest.fixture
    def shape(self):
        return self.base_shape

    @pytest.fixture
    def unitary(self):
        return Mat(*self.base_args)

    @pytest.fixture
    def M1(self, obj):
        return obj

    @pytest.fixture
    def M2(self, other):
        return other

    @pytest.fixture
    def M(self, obj):
        return obj

    # Test matrix constructors
    def test_create_from_cols(self, M):
        cols = list(M.T)
        assert M == Mat.from_cols(cols)

    def test_create_from_rows(self, M):
        rows = list(M)
        assert M == Mat.from_rows(rows)

    def test_create_from_empty_dict(self, zero, shape):
        m, n = shape
        mat = Mat[m, n].from_dict({})
        assert mat == zero

    def test_create_from_dict(self, zero, shape):
        m, n = shape
        mat = Mat[m, n].from_dict({(0, 0): 1})
        assert mat != zero
        assert mat[0, 0] == 1

    # Conversions
    def test_conversion_to_dict(self, M):
        D = M.as_dict()
        assert isinstance(D, dict)
        assert 0 not in D.values()
        assert D
        assert type(M).from_dict(D) == M

    def test_conversion_to_items(self, M):
        D = M.as_dict()
        items = {k: v for k, v in M.items() if v}
        assert D == items

    # Accessors
    def test_col_accessor(self, M):
        col = M.col(0)
        assert isinstance(col, Vec)
        assert col == M.T[0]

    def test_row_accessor(self, M):
        row = M.row(0)
        assert isinstance(row, Vec)
        assert row == M[0]

    # Copy
    def test_copy_changing_terms(self, M):
        assert M == M.copy()

        cp = M.copy(A00=42)
        assert cp[0, 0] == 42
        assert M == cp.copy(A00=M[0, 0])

    # Shape transformations
    def test_append_column(self, M, shape):
        m, n = shape
        col = M.col(0)
        M2 = M.append_col(col)
        assert M2.shape == (m, n + 1)
        assert M2[0, n] == col[0]


class SquareBase(MatrixBase):
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
        M = Mat[N, N].from_diag(diag)
        assert M * M == M

    def test_trace_of_identity(self, I, N):
        return I.trace() == N

    def test_identity_diag(self, I, N):
        return list(I.diag()) == [1] * N

    def test_identity_set_null_diag(self, I, N, zero):
        assert I.set_diag([0] * N) == zero

    def test_identity_drop_diag(self, I, zero):
        assert I.drop_diag() == zero

    @pytest.mark.skip(reason='not implemented yet')
    def test_identity_has_unity_eigenvalues(self, I, N):
        assert I.eigenvalues() == [1] * N

    def test_identity_is_inverse(self, I):
        assert I.inv() == I

    def test_identity_solve(self, I, N):
        b = Vec(*range(1, N + 1))
        assert I.solve(b) == b
        assert I.solve_jacobi(b) == b
        assert I.solve_gauss(b) == b
        assert I.solve_triangular(b) == b


class TestMat2x2(SquareBase):
    base_shape = (2, 2)
    base_args = [1, 2], [3, 4]
    base_args__other = [1, 2], [1, 2]
    base_args__zero = [0, 0], [0, 0]
    base_args__add_ab = [2, 4], [4, 6]
    base_args__sub_ab = [0, 0], [2, 2]
    base_args__I = [1, 0], [0, 1]
    base_args__smul = [2, 4], [6, 8]
    base_args__mul = [3, 6], [7, 14]

    def test_inverse_of_diagonal_matrix(self):
        M = Mat([2, 0], [0, 4])
        Minv = M.inv()
        assert Minv == Mat([0.5, 0], [0, 0.25])

    def test_diagonal_inverse_recovers_identity(self):
        M = Mat([2.0, 0.0], [0.0, 4.0])
        Minv = M.inv()
        assert Minv.__origin__ is M.__origin__
        assert type(Minv) is type(M)
        assert Minv * M == Mat([1.0, 0.0], [0.0, 1.0])
        assert M * Minv == Mat([1.0, 0.0], [0.0, 1.0])

    def test_inverse(self, M, I):
        assert M.inv() * M == I
        assert M * M.inv() == I


class TestMat3x3(SquareBase):
    base_shape = (3, 3)
    base_args = [1, 2, 3], [4, 5, 6], [7, 8, 9]
    base_args__other = [1, 2, 3], [1, 2, 3], [1, 2, 3]
    base_args__zero = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    base_args__add_ab = [2, 4, 6], [5, 7, 9], [8, 10, 12]
    base_args__sub_ab = [0, 0, 0], [3, 3, 3], [6, 6, 6]
    base_args__I = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    base_args__smul = [2, 4, 6], [8, 10, 12], [14, 16, 18]
    base_args__mul = [6, 12, 18], [15, 30, 45], [24, 48, 72]


# Basic matrix construction
def test_make_rect():
    M = Mat([1, 2], [3, 4], [5, 6])
    assert M.shape == (3, 2)
    assert M.ncols == 2
    assert M.nrows == 3


def test_make_rect_t():
    assert Mat([1, 2, 3], [4, 5, 6]).shape == (2, 3)


def test_fromrows_sqr():
    M1 = Mat([1, 2], [3, 4])
    M2 = Mat.from_rows([[1, 2], [3, 4]])
    assert M1 == M2


def test_equal_to_list():
    M1 = Mat([1, 2], [3, 4])
    M2 = [[1, 2], [3, 4]]
    assert M1 == M2


def test_get_item():
    M1 = Mat([1, 2], [3, 4])
    assert M1[0] == [1, 2]
    assert M1[1] == [3, 4]


def test_flat2x2():
    M1 = Mat([1, 2], [3, 4])
    assert list(M1.flat) == [1, 2, 3, 4]


def test_flat2x2_redo():
    M1 = Mat([1, 2], [3, 4])
    assert M1, Mat[2, 2].from_flat(M1.flat)


# Rows and cols manipulations
def test_cols():
    M1 = Mat([1, 2], [3, 4], [5, 6])
    assert list(M1.cols()) == [Vec(1, 3, 5), Vec(2, 4, 6)]


def test_rows():
    M1 = Mat([1, 2], [3, 4], [5, 6])
    assert list(M1.rows()) == [Vec(1, 2), Vec(3, 4), Vec(5, 6)]


def test_withrow_vec():
    M1 = Mat([1, 2], [3, 4])
    M2 = Mat([1, 2], [3, 4], [5, 6])
    M3 = M1.append_row([5, 6])
    assert M2 == M3


def test_withrow_matrix():
    M1 = Mat([1, 2], [3, 4])
    M2 = Mat([1, 2], [3, 4], [5, 6], [7, 8])
    M3 = M1.append_row(Mat([5, 6], [7, 8]))
    assert M2 == M3


def test_withrow_vec_middle():
    M1 = Mat([1, 2], [3, 4])
    M2 = Mat([1, 2], [5, 6], [3, 4])
    M3 = M1.append_row([5, 6], index=1)
    assert M2 == M3


def test_withrow_matrix_middle():
    M1 = Mat([1, 2], [3, 4])
    M2 = Mat([1, 2], [5, 6], [7, 8], [3, 4])
    M3 = M1.append_row(Mat([5, 6], [7, 8]), index=1)
    assert M2 == M3


def test_withcol_vec():
    M1 = Mat([1, 2], [3, 4])
    M2 = Mat([1, 2, 5], [3, 4, 6])
    M3 = M1.append_col([5, 6])
    assert M2 == M3


def test_withcol_vec_middle():
    M1 = Mat([1, 2], [3, 4])
    M2 = Mat([1, 5, 2], [3, 6, 4])
    M3 = M1.append_col([5, 6], index=1)
    assert M2 == M3


def test_droppingrow():
    M1 = Mat([1, 2], [3, 4], [5, 6])
    M2 = Mat([1, 2], [5, 6])
    assert M1.drop_row(1) == (M2, Vec(3, 4))


def test_droppingcol():
    M1 = Mat([1, 2], [3, 4], [5, 6])
    M2 = Mat([1], [3], [5])
    assert M1.drop_col(1) == (M2, Vec(2, 4, 6))


def test_selectrows():
    M1 = Mat([1, 2], [3, 4], [5, 6])
    M2 = Mat([1, 2], [5, 6])
    assert M1.select_rows(0, 2) == M2


def test_selectcols():
    M1 = Mat([1, 2, 3], [4, 5, 6], [7, 8, 9])
    M2 = Mat([1, 3], [4, 6], [7, 9])
    assert M1.select_cols(0, 2) == M2


def test_transpose():
    M1 = Mat([1, 2, 3], [4, 5, 6])
    M2 = Mat([1, 4], [2, 5], [3, 6])
    assert M1.T == M2
    assert M1 == M2.T


def test_transpose_sqr():
    M1 = Mat([1, 2], [3, 4])
    M2 = Mat([1, 3], [2, 4])
    assert M1.T == M2
    assert M1 == M2.T


def test_widthdiag():
    M1 = Mat([1, 2], [3, 4])
    M2 = Mat([0, 2], [3, 0])
    assert M1.set_diag([0, 0]) == M2
    assert M1.drop_diag() == M2


# Linear algebra operations
def test_eig2():
    M1 = Mat([1, 0], [0, 2])
    assert M1.eigenpairs() == [(2.0, (0.0, 1.0)),
                               (1.0, (1.0, 0.0))]


def test_det2():
    M1 = Mat([1, 0], [0, 2])
    M2 = Mat([1, 2], [3, 4])
    assert M1.det() == 2
    assert M2.det() == -2


# Matrix mutation
def test_setvalue():
    M1 = mMat([1, 2], [3, 4])
    M1[0] = [1, 1]
    M1[1] = [2, 2]
    assert list(M1.flat) == [1, 1, 2, 2]


def test_colmul():
    M1 = mMat([1, 2], [3, 4])
    M1.imul_col(1, 2)
    M2 = Mat([1, 4], [3, 8])
    assert M1 == M2


def test_rowmul():
    M1 = mMat([1, 2], [3, 4])
    M1.imul_row(1, 2)
    M2 = Mat([1, 2], [6, 8])
    assert M1 == M2


def test_coladd():
    M1 = mMat([1, 2], [3, 4])
    M1.isum_col(1, (1, 1))
    M2 = Mat([1, 3], [3, 5])
    assert M1 == M2


def test_rowadd_sqr():
    M1 = mMat([1, 2], [3, 4])
    M1.isum_row(1, (1, 1))
    M2 = Mat([1, 2], [4, 5])
    assert M1 == M2


def test_rowadd():
    M1 = mMat([1, 2, 3], [4, 5, 6])
    M1.isum_row(1, (1, 1, 1))
    M2 = mMat([1, 2, 3], [5, 6, 7])
    assert M1 == M2


def test_swaprows():
    M1 = mMat([1, 2], [3, 4])
    M1.iswap_rows(0, 1)
    M2 = Mat([3, 4], [1, 2])
    assert M1 == M2


def test_swapcols():
    M1 = mMat([1, 2], [3, 4])
    M1.iswap_cols(0, 1)
    M2 = Mat([2, 1], [4, 3])
    assert M1 == M2


# Regression testings
def test_rotation_matrix():
    assert Rotation2d(0) == [[1, 0], [0, 1]]
