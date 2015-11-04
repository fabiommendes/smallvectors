import unittest
from smallvectors.tools import shape, commonbase

class CoreTests(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(shape([[1, 2], [3, 4], [5, 6]]), (3, 2))
        
    def test_commonbase(self):
        self.assertEqual(commonbase(int, int), int)
        self.assertEqual(commonbase(int, int, int), int)
        

if __name__ == '__main__':
    unittest.main()