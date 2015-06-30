import owl
import cpumpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
    	base = np.arange(0,10, dtype = np.float32)
    	base = np.reshape(base,[2,5])
    	expected = np.transpose(base)
    	
    	tmp = owl.from_numpy(base)
        test = tmp.trans()
        #print 'Expected\n',expected
        #print "Actual\n",test.to_numpy()
        self.assertTrue(np.allclose(expected,test.to_numpy()))
                        
                        
if __name__ == "__main__":
    unittest.main()
