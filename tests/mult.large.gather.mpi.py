import owl
import mpitestinit
from mpitestinit import devices as d
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        owl.set_device(d[3%len(d)])
        a = owl.ones([1000,900])
        owl.set_device(d[2%len(d)])
        b = owl.ones([900,1000])
        owl.set_device(d[1%len(d)])
        test = a*b
        expected = np.ones([1000,1000])*900
        #print 'Expected\n',expected
        #print "Actual\n",test.to_numpy()
        self.assertTrue(np.array_equal(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
