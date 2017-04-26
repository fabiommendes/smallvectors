from smallvectors import Mat


class TestNonSquareMatrices:
    # Basic matrix construction
    def test_make_rect():
        M = Mat([1, 2], [3, 4], [5, 6])
        assert M.shape == (3, 2)
        assert M.ncols == 2
        assert M.nrows == 3


    def test_make_rect_t():
        assert Mat([1, 2, 3], [4, 5, 6]).shape == (2, 3)


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