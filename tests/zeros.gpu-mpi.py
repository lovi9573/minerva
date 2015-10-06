import owl
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test_ones(self):
	owl.set_device(owl.create_mpi_device(1,1))
        test = 0
	for i in range(1000):
		owl.zeros([10000,10000])
		owl.wait_for_all()
	owl.print_profiler_result()


if __name__ == "__main__":
    unittest.main()
