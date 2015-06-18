#!/usr/bin/env python
import sys
import time
import owl
import owl.conv


if __name__ == "__main__":

    print "CPU creation in rank",owl.rank()
    cpu = owl.create_cpu_device()
    print "First cpu device created."
    sys.stdout.flush()
    print '''
         __   __   _   __   _   _____   ____    _    _   ___
        /  | /  | | | |  \\ | | |  ___| |  _ \\  | |  / / /   |
       /   |/   | | | |   \\| | | |__   | |_| | | | / / / /| |
      / /|   /| | | | |      | |  __|  |    /  | |/ / / /_| |
     / / |  / | | | | | |\\   | | |___  | |\\ \\  |   / / ___  |
    /_/  |_/  |_| |_| |_| \\__| |_____| |_| \\_\\ |__/ /_/   |_|
    '''
    if owl.has_cuda():
        gpu = [owl.create_gpu_device(i) for i in range(owl.get_gpu_device_count())]
        print '[INFO] You have %d GPU devices' % len(gpu)
        print '[INFO] Set device to gpu[0]'
        owl.set_device(gpu[0])
    else:
        print '[INFO] CUDA disabled'
        print '[INFO] Set device to cpu'
        owl.set_device(cpu)
    #import IPython; IPython.start_ipython(argv=[])
