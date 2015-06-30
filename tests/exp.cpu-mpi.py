import owl
import cpumpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        base = np.arange(-5,5)
        t = owl.NArray.from_numpy(base)
        test = owl.NArray.exp(t)
        expected = np.exp(base)
        #print 'Expected\n',expected
        #print "Actual\n",test.to_numpy()
        self.assertTrue(np.allclose(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
