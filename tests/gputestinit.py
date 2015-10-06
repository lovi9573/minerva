import owl

devices = []
devices.append(owl.create_gpu_device(0))
owl.set_device(devices[-1])
