import owl

devices = []
gpu_devices = []

#local
devices.append(owl.create_cpu_device())
if owl.has_cuda():
    n = owl.get_gpu_device_count()
    for i in range(n):
        gpu_devices.append(owl.create_gpu_device(i))

#remote
if owl.has_mpi():
    n = owl.get_mpi_node_count()
    for i in range(1,n):
        id = owl.create_mpi_device(i,0)
        devices.append(id)
        if owl.has_cuda():
            n = owl.get_mpi_device_count()
            for g in range(n):
                gpu_devices.append(owl.create_mpi_device(i,g+1))     
owl.set_device(gpu_devices[-1])