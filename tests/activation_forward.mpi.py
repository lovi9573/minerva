import owl
import mpitestinit
import unittest
import numpy as np
from owl import conv

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        base = np.asarray([10.0,20.0,30.0,40.0])
        owlarray = owl.from_numpy(base)
        expected = base / 100.0
        test = conv.softmax(owlarray)
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        self.assertTrue(np.array_equal(expected, test.to_numpy())))


if __name__ == "__main__":
    unittest.main()
