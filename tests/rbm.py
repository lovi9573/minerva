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
        dev = owl.create_gpu_device()
    else:
        dev = cpu
    owl.set_device(dev)
    
    # load data
    gzfile = gzip.GzipFile('/home/jlovitt/Downloads/mnist.dat','rb')
    #discard stored variable name
    pickle.load(gzfile)
    data = pickle.load(gzfile)
    #data = np.loadtxt(DATAFILE,dtype=np.float32, delimiter=" ")
    data = data - np.mean(data, 0)
    data = data / np.var(data, 0)
    
    # training parameters
    epsilon = 0.01
    momentum = 0.9
    
    num_epochs = 10
    batch_size = 10
    num_batches = data.shape[0]//batch_size
    
    # model parameters
    num_vis = data.shape[1]
    num_hid = 10
    
    # initialize weights
    weights = 0.1 * owl.randn([num_vis, num_hid],0,1)
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
            np_set = data[batch*batch_size:(batch + 1)*batch_size,:]
            training_set = owl.from_numpy(np_set)
            v = training_set
    
            # apply momentum
            d_weights *= momentum
            d_bias_v *= momentum
            d_bias_h *= momentum
    
            # Propogate to hiddens and get positive phase
            hiddens = 1.0 / (1 + owl.NArray.exp(0-((weights.trans()* v) + bias_h)))
    
            d_weights += v* hiddens.trans()
            d_bias_v += v.sum(1)
            d_bias_h += hiddens.sum(1)

            # sample hiddens
            sampled_hiddens = owl.from_numpy(1.0 * (hiddens > np.random.rand(num_hid, batch_size)))
    
            #propogate to visible
            visibles = 1.0 / (1 + owl.NArray.exp(0-((weights* sampled_hiddens) + bias_v)))
    
            # negative phase
            sampled_set = 1. / (1 + owl.NArray.exp(0-((weights.trans()* v) + bias_h)))
    
            d_weights -= (visibles* sampled_set.trans())
            d_bias_v -= visibles.sum(1)
            d_bias_h -= sampled_set.sum(1)
    
            # update weights
            weights += epsilon/batch_size * d_weights
            bias_v += epsilon/batch_size * d_bias_v
            bias_h += epsilon/batch_size * d_bias_h
    
            errs = (visibles - training_set)
            errs = el.mult(errs,errs)
            
            err.append(errs.sum([0,1]).to_numpy()/reduce(add,errs.shape))
            owl.wait_for_all()
    
        print("Mean squared error: %f" % np.mean(err))
        print("Time: %f" % (time.time() - start_time))
    print "Termination"
