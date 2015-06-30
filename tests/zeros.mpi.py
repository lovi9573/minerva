import owl
import mpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test_zeros(self):
        test = owl.zeros([2,3])
        expected = np.zeros([3,2])
        #print 'Expected\n',expected
        #print "Actual\n",test.to_numpy()
        self.assertTrue(np.array_equal(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()