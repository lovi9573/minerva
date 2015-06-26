import owl
import mpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        base = np.arange(1,15)
        t = owl.NArray.from_numpy(base)
        test = t.max(0)
        expected = np.max(base)
        #print 'Expected\n',expected
        #print "Actual\n",test.to_numpy()
        self.assertEqual(expected, test.to_numpy())


if __name__ == "__main__":
    unittest.main()
