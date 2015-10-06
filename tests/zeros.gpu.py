import owl
import gputestinit
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test_ones(self):
        test = 0
	for i in range(1000):
		test=owl.zeros([10000,10000])
		owl.wait_for_all()
	owl.print_profiler_result()


if __name__ == "__main__":
    unittest.main()
