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
        img = np.arange(0,32, dtype=np.float32) #/32
        img = np.reshape(img,[1,2,4,4])
        expected = np.asarray([[[5,7],
                                [13,15]],
                               [[21,23],
                                [29,31]]]) #/32.0
        #expected = np.asarray([[[ 110.25,  124.25],
        #                        [ 166.25,  180.25]],
        #                       [[ 278.25,  324.25],
        #                        [ 462.25,  508.25]]])
        
        # test
        owlimg = owl.from_numpy(img)
        pooler = owl.conv.Pooler(2,2,2,2)   
        test = pooler.ff(owlimg)
        
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        print "This test must be run with a fractional bit width of 12"
        self.assertTrue(np.allclose(expected, test.to_numpy(), atol= 1.0/(1<<12)*4))


if __name__ == "__main__":
    unittest.main()
