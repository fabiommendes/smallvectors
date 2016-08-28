from smallvectors import Mat, mMat, Rotation2d, Vec
from smallvectors.tests.test_arithmetic import *

#
# Arithmetic tests
#
Mat2 = Mat[2, 2, int]
make_arithmetic_fixtures(
    x=Mat2([1, 2], [3, 4]),
    y=Mat2([3, 4], [1, 2]),
    zero=Mat2([0, 0], [0, 0]),
    add=Mat2([4, 6], [4, 6]),
    sub=Mat2([-2, -2], [2, 2]),
    rsub=Mat2([2, 2], [-2, -2]),
    mul=Mat2([5, 8], [13, 20]),
    rmul=Mat2([15, 22], [7, 10]),
    smul=Mat2([2, 4], [6, 8]),
    scalar_add=False,
    ns=globals(),
)


#
# Basic matrix construction
#
def test_make_square():
    assert Mat([1, 2], [3, 4]).shape == (2, 2)


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


def test_fromcols_sqr():
    M1 = Mat([1, 3], [2, 4])
    M2 = Mat.from_cols([[1, 2], [3, 4]])
    assert M1 == M2


def test_fromrows():
    M1 = Mat([1, 2], [3, 4], [5, 6])
    M2 = Mat.from_rows([[1, 2], [3, 4], [5, 6]])
    assert M1 == M2


def test_fromcols():
    M1 = Mat([1, 3, 5], [2, 4, 6])
    M2 = Mat.from_cols([[1, 2], [3, 4], [5, 6]])
    assert M1 == M2


def test_fromcolsT():
    M1 = Mat([0, 3], [1, 4], [2, 5])
    M2 = Mat.from_cols([[0, 1, 2], [3, 4, 5]])
    assert M1 == M2


def test_fromcolrows_sqr():
    M1 = Mat.from_rows([[1, 2], [3, 4]])
    M2 = Mat.from_cols([[1, 2], [3, 4]])
    assert M1.T == M2
    assert M1 == M2.T


#
# Basic properties and behaviors
#
def test_equal():
    M1 = Mat([1, 2], [3, 4])
    M2 = Mat([1, 2], [3, 4])
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


#
# Rows and cols manipulations
#
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


#
# Linear algebra operations
#
def test_eig2():
    M1 = Mat([1, 0], [0, 2])
    assert M1.eigenpairs() == [(2.0, (0.0, 1.0)),
                               (1.0, (1.0, 0.0))]


def test_det2():
    M1 = Mat([1, 0], [0, 2])
    M2 = Mat([1, 2], [3, 4])
    assert M1.det() == 2
    assert M2.det() == -2


def test_inv2_trivial():
    M1 = Mat([1, 0], [0, 2])
    M2 = Mat([1, 0], [0, 0.5])
    assert M1.inv() == M2


def test_inv2():
    M1 = Mat([1, 2], [3, 4])
    M2 = Mat([1, 0], [0, 1])
    assert M1.inv() * M1 == M2
    assert M1 * M1.inv() == M2


#
# Matrix mutation
#
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


#
# Regression testings
#
def test_rotation_matrix():
    assert Rotation2d(0) == [[1, 0], [0, 1]]


if __name__ == '__main__':
    pytest.main('test_matrix.py -q')
