from smallvectors.tests import ArithmeticUnittest, unittest
from smallvectors import Mat, MatAny, mMat, Vec

class MatArithmeticTests(ArithmeticUnittest):
    obj_type = Mat[2, 2, int]
    commutes = ['add']
    
    def names(self):
        Mat = self.obj_type
        
        m = Mat([1, 2], 
                [3, 4])
        
        t = Mat([3, 4], 
                [1, 2])
        r = 2
        k = 0.5

        add_mt = Mat([4, 6], [4, 6])
        sub_mt = Mat([-2, -2], [2, 2])
        sub_tm = Mat([2, 2], [-2, -2])

        mul_rm = Mat([4, 6], [4, 6])
        mul_km = Mat([4, 6], [4, 6])
        div_mr = mul_km 
        div_mk = mul_rm
        
        mul_mt = Mat([5, 8], 
                     [13, 20])
        
        mul_tm = Mat([15, 22], 
                     [7, 10])

        del Mat
        return locals()


class MatTests(unittest.TestCase):
    def test_equal(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat([1, 2], [3, 4])
        self.assertEqual(M1, M2)

    def test_get_item(self):
        M1 = Mat([1, 2], [3, 4])
        self.assertEqual(M1[0], [1, 2])
        self.assertEqual(M1[1], [3, 4])
        
    def test_make_square(self):
        self.assertEqual(Mat([1, 2], [3, 4]).shape, (2, 2))
        
    def test_make_rect(self):
        M = Mat([1, 2], [3, 4], [5, 6])
        self.assertEqual(M.shape, (3, 2))
        self.assertEqual(M.ncols, 2)
        self.assertEqual(M.nrows, 3)
        
    def test_make_rect_T(self):
        self.assertEqual(Mat([1, 2, 3], [4, 5, 6]).shape, (2, 3))
        
    def test_equal_to_list(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = [[1, 2], [3, 4]]
        self.assertEqual(M1, M2)
    
    def test_fromrows_sqr(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat.fromrows([[1, 2], [3, 4]])
        self.assertEqual(M1, M2)

    def test_fromcols_sqr(self):
        M1 = Mat([1, 3], [2, 4])
        M2 = Mat.fromcols([[1, 2], [3, 4]])
        self.assertEqual(M1, M2)
        
    def test_fromrows(self):
        M1 = Mat([1, 2], [3, 4], [5, 6])
        M2 = Mat.fromrows([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(M1, M2)

    def test_fromcols(self):
        M1 = Mat([1, 3, 5], [2, 4, 6])
        M2 = Mat.fromcols([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(M1, M2)
        
    def test_fromcolsT(self):
        M1 = Mat([0, 3], [1, 4], [2, 5])
        M2 = Mat.fromcols([[0, 1, 2], [3, 4, 5]])
        self.assertEqual(M1, M2)
        
    def test_fromcolrows_sqr(self):
        M1 = Mat.fromrows([[1, 2], [3, 4]])
        M2 = Mat.fromcols([[1, 2], [3, 4]])
        self.assertEqual(M1.T, M2)
        self.assertEqual(M1, M2.T)
        
    def test_cols(self):
        M1 = Mat([1, 2], [3, 4], [5, 6])
        self.assertEqual(list(M1.cols()), [Vec(1, 3, 5), Vec(2, 4, 6)])
        
    def test_rows(self):
        M1 = Mat([1, 2], [3, 4], [5, 6])
        self.assertEqual(list(M1.rows()), [Vec(1, 2), Vec(3, 4), Vec(5, 6)])
        
    def test_withrow_vec(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat([1, 2], [3, 4], [5, 6])
        M3 = M1.withrow([5, 6])
        self.assertEqual(M2, M3)
        
    def test_withrow_matrix(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat([1, 2], [3, 4], [5, 6], [7, 8])
        M3 = M1.withrow(Mat([5, 6], [7, 8]))
        self.assertEqual(M2, M3)

    def test_withrow_vec_middle(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat([1, 2], [5, 6], [3, 4])
        M3 = M1.withrow([5, 6], index=1)
        self.assertEqual(M2, M3)

    def test_withrow_matrix_middle(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat([1, 2], [5, 6], [7, 8], [3, 4])
        M3 = M1.withrow(Mat([5, 6], [7, 8]), index=1)
        self.assertEqual(M2, M3)

    def test_withcol_vec(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat([1, 2, 5], [3, 4, 6])
        M3 = M1.withcol([5, 6])
        self.assertEqual(M2, M3)
        
    def test_withcol_vec_middle(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat([1, 5, 2], [3, 6, 4])
        M3 = M1.withcol([5, 6], index=1)
        self.assertEqual(M2, M3)
        
    def test_droppingrow(self):
        M1 = Mat([1, 2], [3, 4], [5, 6])
        M2 = Mat([1, 2], [5, 6])
        self.assertEqual(M1.droppingrow(1), (M2, Vec(3, 4)))

    def test_droppingcol(self):
        M1 = Mat([1, 2], [3, 4], [5, 6])
        M2 = Mat([1], [3], [5])
        self.assertEqual(M1.droppingcol(1), (M2, Vec(2, 4, 6)))

    def test_selectrows(self):
        M1 = Mat([1, 2], [3, 4], [5, 6])
        M2 = Mat([1, 2], [5, 6])
        self.assertEqual(M1.selectrows([0, 2]), M2)

    def test_selectcols(self):
        M1 = Mat([1, 2, 3], [4, 5, 6], [7, 8, 9])
        M2 = Mat([1, 3], [4, 6], [7, 9])
        self.assertEqual(M1.selectcols([0, 2]), M2)

    def test_transpose(self):
        M1 = Mat([1, 2, 3], [4, 5, 6])
        M2 = Mat([1, 4], [2, 5], [3, 6])
        self.assertEqual(M1.T, M2)
        self.assertEqual(M1, M2.T)

    def test_transpose_sqr(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat([1, 3], [2, 4])
        self.assertEqual(M1.T, M2)
        self.assertEqual(M1, M2.T)

    def test_widthdiag(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat([0, 2], [3, 0])
        self.assertEqual(M1.withdiag([0, 0]), M2)
        self.assertEqual(M1.droppingdiag(), M2)
        
    def test_flat2x2(self):
        M1 = Mat([1, 2], [3, 4])
        self.assertEqual(list(M1.flat), [1, 2, 3, 4])

    def test_flat2x2_redo(self):
        M1 = Mat([1, 2], [3, 4])
        self.assertEqual(M1, Mat[2, 2].fromflat(M1.flat))


class LinalgOperationsTests(unittest.TestCase):        
    def test_eig2(self):
        M1 = Mat([1, 0], [0, 2])
        self.assertEqual(M1.eigenpairs(), [(2.0, (0.0, 1.0)), 
                                           (1.0, (1.0, 0.0))])

    def test_det2(self):
        M1 = Mat([1, 0], [0, 2])
        M2 = Mat([1, 2], [3, 4])
        self.assertEqual(M1.det(), 2)
        self.assertEqual(M2.det(), -2)

    def test_inv2_trivial(self):
        M1 = Mat([1, 0], [0, 2])
        M2 = Mat([1, 0], [0, 0.5])
        self.assertEqual(M1.inv(), M2)
        
    def test_inv2(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat([1, 0], [0, 1])
        self.assertEqual(M1.inv() * M1, M2)
        self.assertEqual(M1 * M1.inv(), M2)
        
        
class MutableMatrixTests(unittest.TestCase):
    def test_setvalue(self):
        M1 = mMat([1, 2], [3, 4])
        M1[0] = [1, 1]
        M1[1] = [2, 2]
        self.assertEqual(list(M1.flat), [1, 1, 2, 2])
    
    def test_colmul(self):
        M1 = mMat([1, 2], [3, 4])
        M1.colmul(1, 2)
        M2 = Mat([1, 4], [3, 8])
        self.assertEqual(M1, M2)
        
    def test_rowmul(self):
        M1 = mMat([1, 2], [3, 4])
        M1.rowmul(1, 2)
        M2 = Mat([1, 2], [6, 8])
        self.assertEqual(M1, M2)
        
    def test_coladd(self):
        M1 = mMat([1, 2], [3, 4])
        M1.coladd(1, (1, 1))
        M2 = Mat([1, 3], [3, 5])
        self.assertEqual(M1, M2)
        
    def test_rowadd_sqr(self):
        M1 = mMat([1, 2], [3, 4])
        M1.rowadd(1, (1, 1))
        M2 = Mat([1, 2], [4, 5])
        self.assertEqual(M1, M2)
        
    def test_rowadd(self):
        M1 = mMat([1, 2, 3], [4, 5, 6])
        M1.rowadd(1, (1, 1, 1))
        M2 = mMat([1, 2, 3], [5, 6, 7])
        self.assertEqual(M1, M2)
        
    def test_swaprows(self):
        M1 = mMat([1, 2], [3, 4])
        M1.swaprows(0, 1)
        M2 = Mat([3, 4], [1, 2])
        self.assertEqual(M1, M2)
        
    def test_swapcols(self):
        M1 = mMat([1, 2], [3, 4])
        M1.swapcols(0, 1)
        M2 = Mat([2, 1], [4, 3])
        self.assertEqual(M1, M2)
    
        
    
        
if __name__ == '__main__':
    unittest.main()



