import unittest
from generic.tests.test_parametric import ParametricTestCase
from smallvectors import Vec, VecAny, SmallVectorsBase, SmallVectorsMeta


class ParametricVecTestCase(ParametricTestCase):
    def setUp(self):
        self.A = VecAny
        self.B = Vec

    def test_partial_parameters(self):
        self.assertEqual(SmallVectorsBase.__parameters__, None)
        self.assertEqual(self.B.__parameters__, (int, type))
        self.assertEqual(self.B[2].__parameters__, (2, type))

    def test_correct_metatypes(self):
        self.assertEqual(type(self.A), SmallVectorsMeta)
        self.assertEqual(type(self.B), SmallVectorsMeta)
        self.assertEqual(type(self.B[2, float]), SmallVectorsMeta)

    def test_abstract_has_ndim(self):
        self.assertEqual(self.A.ndim, 1)
    
    def test_origin_has_ndim(self):
        self.assertEqual(self.B.ndim, 1)
        
    def test_concrete_has_ndim(self):
        self.assertEqual(self.B[2, float].ndim, 1)

    def test_has_size(self):
        self.assertEqual(self.A.size, None)
        self.assertEqual(self.B.size, None)
        self.assertEqual(self.B[2, float].size, 2)
        
    def test_abstract_has_shape(self):
        self.assertEqual(self.A.shape, (None,))
    
    def test_origin_has_shape(self):
        self.assertEqual(self.B.shape, (None,))
    
    def test_concrete_has_shape(self):
        self.assertEqual(self.B[2, float].shape, (2,))
        
    def test_has_dtype(self):
        self.assertEqual(self.A.dtype, None)
        self.assertEqual(self.B.dtype, None)
        self.assertEqual(self.B[2, float].dtype, float)


if __name__ == '__main__':
    unittest.main()