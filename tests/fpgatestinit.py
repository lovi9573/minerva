import owl

devices = []
devices.append(owl.create_cpu_device())
id = owl.create_fpga_device(0)
devices.append(id)
owl.set_device(devices[0])

def setfpga():
    owl.set_device(devices[-1])