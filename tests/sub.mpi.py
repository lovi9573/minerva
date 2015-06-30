import owl
import mpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        a = owl.ones([20,900])
        b = owl.ones([20,900])
        testa = a-b
        testb = a - 1
        expected = np.zeros([900,20])
        #print 'Expected\n',expected
        #print "Actuala\n",testa.to_numpy()
        #print "Actualb\n",testb.to_numpy()
        self.assertTrue(np.array_equal(expected, testa.to_numpy()))
        self.assertTrue(np.array_equal(expected, testb.to_numpy()))


if __name__ == "__main__":
    unittest.main()
