import owl
import mpitestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        expected = np.arange(-5,5, dtype=np.float32)
        expected = np.reshape(expected, [2,5])
        test = owl.NArray.from_numpy(expected)
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        self.assertTrue(np.array_equal(expected, test.to_numpy()))


if __name__ == "__main__":
    pass
    unittest.main()
