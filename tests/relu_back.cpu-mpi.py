import owl
import cpumpitestinit
import unittest
import numpy as np
from owl import elewise

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        bottom = np.asarray([2,-1,0,1,2,3], np.float32)
        top = np.asarray([0,0,0,1,2,3], np.float32)
        top_diff = np.asarray([0.1,0.1,0.1,0.1,0.1,0.1], np.float32)
        print top_diff.shape
        expected = np.asarray([0,0,0,0.1,0.1,0.1], np.float32)
        owldiff = owl.from_numpy(top_diff)
        owltop = owl.from_numpy(top)
        test = elewise.relu_back(owldiff,owltop)
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        self.assertTrue(np.allclose(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
