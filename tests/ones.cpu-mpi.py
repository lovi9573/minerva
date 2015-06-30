import owl
import cpumpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test_ones(self):
        test = owl.ones([20,1])
        expected = np.ones([1,20])
        #print 'Expected\n',expected
        #print "Actual\n",test.to_numpy()
        self.assertTrue(np.array_equal(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()