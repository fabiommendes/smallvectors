import unittest
from smallvectors.tools import flatten

class ToolsTests(unittest.TestCase):
    def test_flatten(self):
        self.assertEqual(flatten([1, 2]), ([1, 2], 2))
        self.assertEqual(flatten([[1, 2]]), ([1, 2], 1, 2))
        self.assertEqual(flatten([[1], [2]]), ([1, 2], 2, 1))
        self.assertEqual(flatten([[1, 2], [3, 4], [5, 6]]), ([1, 2, 3, 4, 5, 6], 3, 2))
        

if __name__ == '__main__':
    unittest.main()