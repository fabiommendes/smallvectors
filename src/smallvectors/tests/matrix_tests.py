from smallvectors.tests import ArithmeticUnittest, unittest
from smallvectors import Mat, MatAny, mMat

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
        
    def test_equal_to_list(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = [[1, 2], [3, 4]]
        self.assertEqual(M1, M2)
    
    def test_fromrows(self):
        M1 = Mat([1, 2], [3, 4])
        M2 = Mat.fromrows([[1, 2], [3, 4]])
        self.assertEqual(M1, M2)



if __name__ == '__main__':
    unittest.main()


