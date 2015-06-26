import owl

devices = []
devices.append(owl.create_cpu_device())
if owl.has_mpi():
    n = owl.get_mpi_node_count()
    for i in range(1,n):
        id = owl.create_mpi_device(i,0)
        devices.append(id)
owl.set_device(devices[-1])