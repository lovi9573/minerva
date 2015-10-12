import owl
import numpy as np
import owl.elewise as el
import time
import pickle
import gzip
from operator import add

#The file containing the data in the format of one vector per line space separated floats
DATAFILE = "????"



if __name__ == "__main__":
    #Setup minerva
    cpu = owl.create_cpu_device()
    if owl.get_gpu_device_count() > 0:
        dev = owl.create_gpu_device(0)
    else:
        dev = cpu
    owl.set_device(dev)
    
    # load data
    gzfile = gzip.GzipFile('/home/jlovitt/storage/mnist/mnist.dat','rb')
    #discard stored variable name
    pickle.load(gzfile)
    data = pickle.load(gzfile)
    #data = np.loadtxt(DATAFILE,dtype=np.float32, delimiter=" ")
    data = data - np.mean(data, 0)
    data = data / np.var(data, 0)
    
    # training parameters
    epsilon = 0.01
    momentum = 0.9
    
    num_epochs = 40
    batch_size = 64
    num_batches = data.shape[0]//batch_size
    
    # model parameters
    num_vis = data.shape[1]
    num_hid = 1024
    
    # initialize weights
    weights = 0.1 * owl.randn([num_vis, num_hid],0,1)
    bias_v = owl.zeros([1,num_vis])
    bias_h = owl.zeros([1,num_hid])
    
    # initialize weight updates
    d_weights = owl.zeros((num_vis,num_hid ))
    d_bias_v = owl.zeros([1,num_vis])
    d_bias_h = owl.zeros([1,num_hid])
    
    start_time = time.time()
    for epoch in range(num_epochs):
        print("Epoch %i" % (epoch + 1))
        err = []
    
        for batch in range(num_batches):
            np_set = data[batch*batch_size:(batch + 1)*batch_size,:]
            training_set = owl.from_numpy(np_set).trans()
            
    
            # apply momentum
            d_weights *= momentum
            d_bias_v *= momentum
            d_bias_h *= momentum
    
            # Propogate to hiddens 
            #print "=> h"
            z_h = 0-(training_set * weights + bias_h)
            hiddens = 1.0 / (1 + owl.NArray.exp(z_h))
    
            #Get positive Phase
            #print "+ phase"
            d_weights += training_set.trans()* hiddens
            d_bias_v += training_set.sum(0)
            d_bias_h += hiddens.sum(0)

            # sample hiddens
            #print "sample h"
            sampled_hiddens = owl.from_numpy(1.0 * (hiddens.to_numpy() > np.random.rand(num_hid,  batch_size)))
    
            #Reconstruct visible
            #print "v <= "
            z_v = 0-(sampled_hiddens*weights.trans() + bias_v)
            reconstruction = 1.0 / (1 + owl.NArray.exp(z_v))
    
            # Propogate to hiddens 
            #print "=> h"
            z_h = 0-(reconstruction * weights + bias_h)
            hiddens = 1.0 / (1 + owl.NArray.exp(z_h))
    
            #Get negative Phase
            #print "- phase"
            d_weights -= reconstruction.trans()* hiddens
            d_bias_v -= reconstruction.sum(0)
            d_bias_h -= hiddens.sum(0)
    
            # update weights
            #print "update"
            weights += epsilon/batch_size * d_weights
            bias_v += epsilon/batch_size * d_bias_v
            bias_h += epsilon/batch_size * d_bias_h
    
            #Compute errors
            #print "compute errors"
            errs = (reconstruction - training_set)
            errs = el.mult(errs,errs)
            
            #owl.set_device(cpu)
            tmp = errs.sum(0)
            tmp2 = tmp.sum(0)
            err.append(tmp2.to_numpy()/reduce(add,errs.shape))
            owl.wait_for_all()
            #owl.set_device(dev)
    
        print("Mean squared error: %f" % np.mean(err))
        print("Time: %f" % (time.time() - start_time))
    print "Termination"
