import owl
import cputestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
	base = np.arange(-10,10,dtype=np.float32)/2
        a = owl.from_numpy(base)
        test = owl.NArray.relu(a)
        expected = base.clip(0,20)
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        self.assertTrue(np.array_equal(expected,test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
