import owl
import unittest
import numpy as np

class TestForiegnOps(unittest.TestCase):
    
    def test_ones(self):
        t = owl.ones([2,3])
        self.assertTrue(np.array_equal(np.ones([2,3]), t))



if __name__ == "__main__":
    devices = []
    devices.append(owl.create_cpu_device())
    if owl.has_mpi():
        n = owl.get_mpi_node_count()
        for i in range(1,n):
            id = owl.create_mpi_device(i,0)
            devices.append(id)
    owl.set_device(1)
    unittest.main()
    
    