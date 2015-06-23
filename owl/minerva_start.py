#!/usr/bin/env python
import sys
import time
import owl
import owl.conv


if __name__ == "__main__":
    if owl.rank() !=0:
        print "rank ",owl.rank()," entering mpi main loop"
        owl.worker_run()


    cpu = owl.create_cpu_device()
    print "owl: local CPU creation in rank {} with id {}".format(owl.rank(), cpu)
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
    if owl.has_mpi():
        n = owl.get_mpi_node_count()
        for i in range(1,n):
            id = owl.create_mpi_device(i,0)
            print "owl: created mpi cpu device on rank {} with id {}".format(i, id)
    else:
        print '[INFO] CUDA disabled'
        print '[INFO] Set device to cpu'
        owl.set_device(cpu)
    print "\nREADY FOR INPUT\n"
    owl.set_device(1)
    x = owl.ones([2,3])
    y = owl.ones([2,3])
    z = x + y
    #print z.to_numpy()
    #import IPython; IPython.start_ipython(argv=[])
