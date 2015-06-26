import owl
import mpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        diff = owl.ones([20,1])
        test = owl.NArray.activation_backward(diff,diff,diff,activation_algo.sigmoid())
        expected = np.ones([1,20])
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
