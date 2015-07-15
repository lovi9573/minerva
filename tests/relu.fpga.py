import owl
import fpgatestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test(self):
        base = np.arange(-10,10)
        owlbase = owl.from_numpy(base)
        print fpgatestinit.devices
        fpgatestinit.setfpga()
        test = owl.NArray.relu(owlbase)
        expected = base.clip(0,20)
        print 'Expected\n',expected
        print "Actual\n",test.to_numpy()
        self.assertTrue(np.array_equal(expected,test.to_numpy()))


if __name__ == "__main__":
    unittest.main()
