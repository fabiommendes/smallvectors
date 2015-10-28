from smallvectors.tests import unittest
from smallvectors.core import Flat, mFlat, FlatView

class FlatViewTests(unittest.TestCase):
    def setUp(self):
        class WithView:
            size = 4
            
            def __init__(self):
                self.data = [1, 2, 3, 4]
                self.flat = FlatView(self)
                
            def __flatiter__(self):
                return iter(self.data)
            
            def __flatgetitem__(self, i):
                return self.data[i]
                
        self.withview_class = WithView
        self.withview = WithView()
        
    def test_view_getitem(self):
        self.assertEqual(self.withview.flat[0], 1)
        
    def test_view_getslice(self):
        self.assertEqual(self.withview.flat[0:2], [1, 2])
        
    def test_view_getslice_step(self):
        self.assertEqual(self.withview.flat[0::2], [1, 3])


if __name__ == '__main__':
    unittest.main()