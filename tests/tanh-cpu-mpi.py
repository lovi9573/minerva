import owl
import owl.elewise
import cpumpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        base = np.arange(0,100)/10
        base = np.reshape(base,[5,5,4])
        expected = np.tanh(base)
        
        test = owl.from_numpy(base)
        test = owl.elewise.tanh(test)
        #print 'Expected\n',expected
        #print "Actual\n",test.to_numpy()
        self.assertTrue(np.allclose(expected,test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
