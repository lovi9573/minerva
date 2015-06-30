import owl
import cpumpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        a = owl.ones([20,1])
        test = owl.NArray.relu(a)
        expected = np.ones([1,20])
        #print 'Expected\n',expected
        #print "Actual\n",test.to_numpy()
        self.assertTrue(np.array_equal(expected,test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
