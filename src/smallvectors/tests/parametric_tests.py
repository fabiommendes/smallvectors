import unittest
from smallvectors.core import Parametric, ParametricMeta
from smallvectors.core import SmallVectorsMeta, SmallVectorsBase
from smallvectors import Vec, VecAny 
    
class ParametricTests(unittest.TestCase):
    def setUp(self):
        class A(Parametric):
            __abstract__ = True
            __parameters__ = [int, type]
            
        class B(A):
            pass
        
        self.A = A
        self.B = B

    def test_abstract_parameters(self):
        self.assertEqual(self.A.__parameters__ , (int, type))
    
    def test_origin_parameters(self):
        self.assertEqual(self.B.__parameters__ , (int, type))
        
    def test_concrete_parameters(self):
        self.assertEqual(self.B[2, float].__parameters__ , (2, float))        
    
    def test_abstract(self):
        self.assertEqual(self.A.__abstract__, True)
        self.assertEqual(self.A.__origin__, None)
        assert isinstance(self.A.__subtypes__, dict)
        
    def test_sub_abstract(self):
        T = self.A[2, float]
        self.assertEqual(T.__abstract__, True)
        self.assertEqual(T.__origin__, self.A)
        self.assertEqual(T.__subtypes__, None)
        
    def test_origin(self):
        self.assertEqual(self.B.__abstract__, False)
        self.assertEqual(self.B.__origin__, None)
        assert isinstance(self.B.__subtypes__, dict)
    
    def test_concrete_not_abstract(self):
        self.assertEqual(self.B[2, float].__abstract__, False)
        
    def test_concrete_origin(self):
        self.assertEqual(self.B[2, float].__origin__, self.B)
        self.assertEqual(self.B[2, float].__subtypes__, None)
        
    def test_partial_parameters(self):
        self.assertEqual(Parametric.__parameters__, None)
        self.assertEqual(self.B.__parameters__, (int, type))
        self.assertEqual(self.B[2].__parameters__, (2, type))

    def test_no_concrete_parametrization(self):
        def test():
            self.B[2, float][2, int]
        self.assertRaises(TypeError, test)
        
    def test_origin_parametrization(self):
        assert self.B[2, float] is self.B[2, float]
        
    def test_abstract_parametrization(self):
        assert self.A[2, float] is self.A[2, float]

    def test_no_abstract_instantiation(self):
        def test():
            self.A(1, 2)
        self.assertRaises(TypeError, test)

    def test_no_parametric_abstract_instantiation(self):
        def test():
            self.A[2, float](1, 2)
        self.assertRaises(TypeError, test)

    def test_correct_metatypes(self):
        self.assertEqual(type(self.A), ParametricMeta)
        self.assertEqual(type(self.B), ParametricMeta)
        self.assertEqual(type(self.B[2, float]), ParametricMeta)
        
    def test_concrete_params(self):
        self.assertEqual(self.B[2, float].__parameters__, (2, float))
        self.assertEqual(self.A[2, float].__parameters__, (2, float))

    def test_partial_params(self):
        self.assertEqual(self.B[2].__parameters__, (2, type))
        self.assertEqual(self.B[2, None], self.B[2])
        self.assertEqual(self.B[2, ...], self.B[2])

    def test_abstract_from_origin(self):
        self.assertEqual(self.B[2].__abstract__, True)
        self.assertEqual(self.B[2, None].__abstract__, True)
        
    def test_abstract_ellipsis(self):
        self.assertEqual(self.B[2, ...].__abstract__, True)
        self.assertEqual(self.B[..., float].__abstract__, True)
        
    def concrete_subclass(self):
        class B2(self.B[2, float]):
            pass
        self.assertEqual(B2(1, 2), (1, 2))


class ParametricVecTests(ParametricTests):
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

#del ParametricVecTests

if __name__ == '__main__':
    unittest.main()