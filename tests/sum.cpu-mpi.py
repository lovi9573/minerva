import owl
import cpumpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        t = owl.ones([20,2])
        testa = t.sum(0)
        testb = t.sum(1)
        expecteda = np.ones([2,1])*20
        expectedb = np.ones([1,20])*2
        #print 'Expecteda\n',expecteda
        #print 'Expectedb\n',expectedb
        #print "Actuala\n",testa.to_numpy()
        #print "Actualb\n",testb.to_numpy()
        self.assertTrue(np.array_equal(expecteda, testa.to_numpy()))
        self.assertTrue(np.array_equal(expectedb, testb.to_numpy()))


if __name__ == "__main__":
    unittest.main()
