import owl
import mpitestinit
import unittest
import numpy as np
from owl import conv

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        base = np.arange(0,100)
        base = np.reshape(base, [10,10,1,1])
        expected = np.asarray([[11,13,15,17,19],
                               [31,33,35,37,39],
                               [51,53,55,57,59],
                               [71,73,75,77,79],
                               [91,93,95,97,99]])
        pooler = conv.Pooler(2,2,2,2)
        #algo = pooling_algo.max()
        owlarray = owl.from_numpy(base)
        test = pooler.ff(owlarray)
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        self.assertTrue(np.array_equal(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
