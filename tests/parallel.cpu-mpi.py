import owl
import math
import cpumpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        narrays = []
        n = 32
        exp = 0
        for i in range(n):
            narrays.append(owl.ones([10,10]))
        j = 1
        while j <= n/2:
            for i in range(0, n , j*2):
                owl.set_device(hash(i)%len(cpumpitestinit.devices))
                narrays[i] = narrays[i]*narrays[i+j]
            j *= 2
            exp = exp*2+1
        test = narrays[0]
        expected = np.ones([10,10])*math.pow(10,exp)
        #print 'Expected\n',expected
        #print "Actual\n",test.to_numpy()
        self.assertTrue(np.allclose(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
