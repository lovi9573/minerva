import owl
import mpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        a = owl.ones([20,2])
        b = owl.ones([20,2])
        testa = owl.NArray.concat([a,b],0)
        #testa = t.sum(0)
        #testb = t.sum(1)
        #expecteda = np.ones([2,1])*20
        expected = np.ones([2,40])*2
        print 'Expected\n',expected
        #print 'Expectedb\n',expectedb
        print "Actual\n",testa.to_numpy()
        #print "Actualb\n",testb.to_numpy()
        self.assertTrue(np.array_equal(expected, testa.to_numpy()))
        #self.assertTrue(np.array_equal(expectedb, testb.to_numpy()))


if __name__ == "__main__":
    unittest.main()
