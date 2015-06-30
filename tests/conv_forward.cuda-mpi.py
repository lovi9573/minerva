import owl
import owl.conv
import cudampitestinit
import unittest
import numpy as np
from scipy.ndimage.filters import convolve

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        # Expected
        in_channels = 3
        in_dim = 11
        out_channels = 5
        out_dim = (in_dim/2 + 1)
        img = np.arange(0,in_dim*in_dim*in_channels*1, dtype=np.float32)
        img = np.reshape(img,[in_dim,in_dim,in_channels,1])
        filter = np.arange(0,3*3*in_channels*out_channels, dtype=np.float32)
        filter = np.reshape(filter,[3,3,in_channels,out_channels])
        bias = np.zeros([5])
        expected = np.zeros([out_dim,out_dim,out_channels])
        for och in range(out_channels):
            tmp = np.zeros([out_dim,out_dim,1])
            for ich in range(in_channels):
                imgslice = np.reshape(img[:,:,ich,0],[in_dim,in_dim])
                filterslice = np.reshape(filter[:,:,ich,och],[3,3])
                tmp += np.reshape(convolve(imgslice,filterslice,mode='constant',cval = 0.0)[::2,::2] , [out_dim, out_dim, 1])
            expected[:,:,och] = np.squeeze(tmp) + bias[och]
            
        # test
        owlimg = owl.from_numpy(np.transpose(img))
        owlfilter = owl.from_numpy(np.transpose(filter))
        owlbias = owl.from_numpy(bias)
        convolver = owl.conv.Convolver(1,1,2,2)   
        test = convolver.ff(owlimg, owlfilter, owlbias)
        
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        self.assertTrue(np.allclose(expected, test))


if __name__ == "__main__":
    unittest.main()
