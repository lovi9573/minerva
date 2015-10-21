import owl
import numpy as np
import owl.elewise as el
import time
import pickle
import gzip
from operator import mul
import matplotlib.pyplot as plt

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
    #data = data - np.mean(data, 0)
    #data = data / np.var(data, 0)
    data = data/255.0
    
    # training parameters
    epsilon = 0.01
    momentum = 0.9
    
    num_epochs = 20
    batch_size = 64
    num_batches = data.shape[1]//batch_size
    
    # model parameters
    num_vis = data.shape[0]
    num_hid = 128
    
    # initialize weights
    np.random.seed(1234)
    weights = owl.from_numpy(0.1 * np.random.randn(num_vis, num_hid)).trans()
    #weights = 0.1 * owl.randn([num_vis, num_hid],0,1)
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
        weights_old = weights
        for batch in range(num_batches):
            np_set = data[:,batch*batch_size:(batch + 1)*batch_size]
            training_set = owl.from_numpy(np_set)
            
    
            # apply momentum
            d_weights *= momentum
            d_bias_v *= momentum
            d_bias_h *= momentum
    
            # Propogate to hiddens 
            #Note: it is ok here to use the probability vector v instead of sampling from v
            #print "=> h"
            z_h = 0-(training_set * weights + bias_h)
            hiddens = 1.0 / (1 + owl.NArray.exp(z_h))
    
            #Get positive Phase
            #Note: Using both probability vectors v and h reduces sampling noise since we are only approximating the convergence of vh of the model.
            #print "+ phase"
            d_weights += training_set.trans()* hiddens
            d_bias_v += training_set.sum(0)
            d_bias_h += hiddens.sum(0)

            # sample hiddens
            #Note: Sampling here is important since the hiddens are being driven by data and we don't want to overfit.
            #print "sample h"
            sampled_hiddens = owl.from_numpy(1.0 * (hiddens.to_numpy() > np.random.rand(num_hid,  batch_size)))
    
            #Reconstruct visible
            #print "v <= "
            z_v = 0-(sampled_hiddens*weights.trans() + bias_v)
            reconstruction = 1.0 / (1 + owl.NArray.exp(z_v))
    
            # Propogate to hiddens 
             #Note: it is ok here to use the probability vector v instead of sampling from v
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
            owl.wait_for_all()
            diff = reconstruction - training_set
            sqrdiff = el.mult(diff,diff)
            sum = sqrdiff.sum([0,1]).to_numpy()[0,0]
            mean =  sum /reduce(mul,sqrdiff.shape)
            err.append(mean)
            #owl.set_device(dev)
    
        print("Mean squared error: %f" % np.mean(err))
        print("Time: %f" % (time.time() - start_time))
        plt.hist((weights - weights_old).to_numpy().flatten(),10)
        plt.show()
        
        im = np.zeros([28,28*num_hid])
        for h in range(num_hid):
            im[:,h*28:(h+1)*28] = weights.to_numpy()[h,:].reshape([28,28])
        plt.hist(weights.to_numpy().flatten(),10)
        plt.show()
        plt.imshow(im, interpolation="none" )
        plt.set_cmap('gray')
        plt.show()
    print "Termination"
