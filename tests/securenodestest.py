import owl
import owl.conv

cpu = owl.create_cpu_device()
if owl.has_cuda():
    gpu = [owl.create_gpu_device(i) for i in range(owl.get_gpu_device_count())]
    print '[INFO] You have %d GPU devices' % len(gpu)
    print '[INFO] Set device to gpu[0]'
    owl.set_device(gpu[0])
else:
    gpu =[]
    print '[INFO] CUDA disabled'
    print '[INFO] Set device to cpu'
    owl.set_device(cpu)

if owl.has_mpi():
    mpi = [[cpu]+gpu]
    for node in range(1,owl.get_mpi_node_count()):
         mpi.append([owl.create_mpi_device(node,i) for i in range(0,owl.get_mpi_device_count(node)+1)])
    print "[INFO] Devices created"
    print mpi
    print '''
================
      PASS
================
'''
else:
    print "[INFO] No mpi support found"
    print '''
================
      FAIL
================
'''



