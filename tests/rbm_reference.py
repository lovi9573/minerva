from __future__ import division
import time
import numpy as np
import scipy.io as scio
import pickle
import gzip
import matplotlib.pyplot as plt

# load data

gzfile = gzip.GzipFile('/home/jlovitt/storage/mnist/mnist.dat','rb')
#discard stored variable name
pickle.load(gzfile)
dat = pickle.load(gzfile)
dat = dat/255.

# training parameters
epsilon = 0.01
momentum = 0.9

num_epochs = 20
batch_size = 64
num_batches = dat.shape[1]//batch_size

# model parameters
num_vis = dat.shape[0]
num_hid = 15

# initialize weights
np.random.seed(1234)
w_vh = 0.1 * np.random.randn(num_vis, num_hid)
w_v = np.zeros((num_vis, 1))
w_h = np.zeros((num_hid, 1))

# initialize weight updates
wu_vh = np.zeros((num_vis, num_hid))
wu_v = np.zeros((num_vis, 1))
wu_h = np.zeros((num_hid, 1))

plt.hist(dat.flatten(),10)
plt.show()
start_time = time.time()
for epoch in range(num_epochs):
    print("Epoch %i" % (epoch + 1))
    err = []
    w_vh_last= w_vh
    for batch in range(num_batches):
        v_true = 0*dat[:, batch*batch_size:(batch + 1)*batch_size]
        v = v_true

        # apply momentum
        wu_vh *= momentum
        wu_v *= momentum
        wu_h *= momentum

        # positive phase
        h = 1. / (1 + np.exp(-(np.dot(w_vh.T, v) + w_h)))

        wu_vh += np.dot(v, h.T)
        wu_v += v.sum(1)[:, np.newaxis]
        wu_h += h.sum(1)[:, np.newaxis]

        # sample hiddens
        h = 1. * (h > np.random.rand(num_hid, batch_size))

        # negative phase
        v = 1. / (1 + np.exp(-(np.dot(w_vh, h) + w_v)))
        h = 1. / (1 + np.exp(-(np.dot(w_vh.T, v) + w_h)))
        
        #print v[0,0]

        wu_vh -= np.dot(v, h.T)
        wu_v -= v.sum(1)[:, np.newaxis]
        wu_h -= h.sum(1)[:, np.newaxis]

        # update weights

        w_vh += epsilon/batch_size * wu_vh
        w_v += epsilon/batch_size * wu_v
        w_h += epsilon/batch_size * wu_h
        
        #print (v - v_true)

        err.append(np.mean((v - v_true)**2))

    print("Mean squared error: %f" % np.mean(err))
    print("Time: %f" % (time.time() - start_time))
    plt.hist((w_vh - w_vh_last).flatten(),10)
    plt.show()
    
    im = np.zeros([28,28*num_hid])
    for h in range(num_hid):
        im[:,h*28:(h+1)*28] = w_vh[:,h].reshape([28,28])
    plt.hist(w_vh.flatten(),10)
    plt.show()
    plt.imshow(im, interpolation="none" )
    plt.set_cmap('gray')
    plt.show()