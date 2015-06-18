import owl
import owl.conv

cpu = owl.create_cpu_device()
if owl.has_cuda():
    gpu = [owl.create_gpu_device(i) for i in range(owl.get_gpu_device_count())]
    print '[INFO] You have %d GPU devices' % len(gpu)
    print '[INFO] Set device to gpu[0]'
    owl.set_device(gpu[0])
else:
    print '[INFO] CUDA disabled'
    print '[INFO] Set device to cpu'
    owl.set_device(cpu)
x = owl.ones([2,10])
y = owl.ones([10,20])
z = x*y
print z.to_numpy()

print '''
================
      PASS
================
'''
