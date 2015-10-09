import owl
import numpy as np
from __future__ import division
import time
import util

#The file containing the data in the format of one vector per line space separated floats
DATAFILE = "????"



if __name__ == "__main__":
    #Setup minerva
    cpu = owl.create_cpu_device()
    if owl.get_num_devices() > 0:
        dev = owl.create_gpu_device()
    else:
        dev = cpu
    owl.set_device(dev)
    
    # load data
    data = np.loadtxt(DATAFILE,dtype=np.float32, delimiter=" ")
    data = data - np.mean(data, 0)
    data = data / np.var(data, 0)
    
    # training parameters
    epsilon = 0.01
    momentum = 0.9
    
    num_epochs = 1
    batch_size = 10
    num_batches = data.shape[0]//batch_size
    
    # model parameters
    num_vis = data.shape[1]
    num_hid = 10
    
    # initialize weights
    weights = 0.1 * owl.randn(num_vis, num_hid)
    bias_v = owl.zeros((num_vis, 1))
    bias_h = owl.zeros((num_hid, 1))
    
    # initialize weight updates
    d_weights = owl.zeros((num_vis, num_hid))
    d_bias_v = owl.zeros((num_vis, 1))
    d_bias_h = owl.zeros((num_hid, 1))
    
    start_time = time.time()
    for epoch in range(num_epochs):
        print("Epoch %i" % (epoch + 1))
        err = []
    
        for batch in range(num_batches):
            training_set = data[batch*batch_size:(batch + 1)*batch_size,:]
            v = training_set
    
            # apply momentum
            d_weights *= momentum
            d_bias_v *= momentum
            d_bias_h *= momentum
    
            # Propogate to hiddens and get positive phase
            hiddens = 1.0 / (1 + np.exp(-(np.dot(weights.T, v) + bias_h)))
    
            # sample hiddens
            sampled_hiddens = 1.0 * (hiddens > np.random.rand(batch_size, num_hid))
    
            #propogate to visible
            visibles = 1.0 / (1 + np.exp(-(np.dot(weights, sampled_hiddens) + bias_v)))
    
            # negative phase
            sampled_set = 1. / (1 + np.exp(-(np.dot(weights.T, v) + bias_h)))
    
            d_weights -= np.dot(v, sampled_set.T)
            d_bias_v -= v.sum(1)[:, np.newaxis]
            d_bias_h -= sampled_set.sum(1)[:, np.newaxis]
    
            # update weights
            weights += epsilon/batch_size * d_weights
            bias_v += epsilon/batch_size * d_bias_v
            bias_h += epsilon/batch_size * d_bias_h
    
            err.append(np.mean((v - training_set)**2))
    
        print("Mean squared error: %f" % np.mean(err))
        print("Time: %f" % (time.time() - start_time))
            d_weights += np.dot(v, E_p.T)
            d_bias_v += v.sum(1)[:, np.newaxis]
            d_bias_h += E_p.sum(1)[:, np.newaxis]
