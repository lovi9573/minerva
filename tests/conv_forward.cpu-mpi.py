import owl
import owl.conv
import cpumpitestinit
import unittest
import numpy as np
from scipy.ndimage.filters import convolve

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        # Expected
        img = np.arange(0,32, dtype=np.float32)
        img = np.reshape(img,[4,4,2,1])
        filter = np.arange(0,2*2*2*2, dtype=np.float32)
        filter = np.reshape(filter,[2,2,2,2])
        bias = np.zeros([2])
        expected = np.asarray([[[1,2],
                                [3,4]],
                               [[1,2],
                                [3,4]]])
        
        # test
        print img
        print filter
        print bias
        owlimg = owl.from_numpy(np.transpose(img))
        owlfilter = owl.from_numpy(np.transpose(filter))
        owlbias = owl.from_numpy(bias)
        convolver = owl.conv.Convolver(0,0,2,2)   
        test = convolver.ff(owlimg, owlfilter, owlbias)
        
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        self.assertTrue(np.allclose(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
