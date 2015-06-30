import owl
import mpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        owl.set_device(mpitestinit.devices[-3])
        a = owl.ones([20,900])
        owl.set_device(mpitestinit.devices[-2])
        b = owl.ones([900,800])
        owl.set_device(mpitestinit.devices[-1])
        test = a*b
        expected = np.ones([800,20])*900
        #print 'Expected\n',expected
        #print "Actual\n",test.to_numpy()
        self.assertTrue(np.array_equal(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
