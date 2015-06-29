import owl
import mpitestinit
import unittest
import numpy as np
from owl import conv

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        base = np.asarray([40.0,20.0,30.0,10.0])
        max = np.max(base)
        base = np.reshape(base, [1,1,1,4])
        owlarray = owl.from_numpy(base)
        expected = np.exp(base - max)
        expected = expected / np.sum(expected)
        test = conv.softmax(owlarray)
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        self.assertTrue(np.allclose(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()