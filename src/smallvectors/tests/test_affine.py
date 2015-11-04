from smallvectors.tests import unittest
from smallvectors import Affine


class AffineTests(unittest.TestCase):
    def test_affine_shape(self):
        T = Affine[2, float]
        self.assertEqual(T.size, 6)
        
if __name__ == '__main__':
    unittest.main()



