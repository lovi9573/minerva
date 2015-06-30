import owl
from owl import elewise
import cpumpitestinit
import unittest
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        base = np.arange(-15,15)
        t = owl.NArray.from_numpy(base)
        test = elewise.sigm(t)
        expected = sigmoid(base)
        #print 'Expected\n',expected
        #print "Actual\n",test.to_numpy()
        self.assertTrue(np.allclose(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
