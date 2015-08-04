import owl
import owl.conv
import unittest
import numpy as np
from scipy.ndimage.filters import convolve

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        # Expected
        cpu=owl.create_cpu_device()
        owl.set_device(cpu)
        img = np.arange(0,32, dtype=np.float32)/32
        img = np.reshape(img,[1,2,4,4])
        filter = np.arange(0,2*2*2*2, dtype=np.float32)/32
        filter = np.reshape(filter,[2,2,2,2])
        bias = np.zeros([2])
        expected = np.asarray([[[441,497],
                                [665,721]],
                               [[1113,1297],
                                [1849,2033]]])/1024.0
        #expected = np.asarray([[[ 110.25,  124.25],
        #                        [ 166.25,  180.25]],
        #                       [[ 278.25,  324.25],
        #                        [ 462.25,  508.25]]])
        
        # test
        owlimg = owl.from_numpy(img)
        owlfilter = owl.from_numpy(filter)
        owlbias = owl.from_numpy(bias)
        convolver = owl.conv.Convolver(0,0,2,2)   
        test = convolver.ff(owlimg, owlfilter, owlbias)
        
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        self.assertTrue(np.allclose(expected, test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
