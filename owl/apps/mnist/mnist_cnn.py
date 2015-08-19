import sys
import time
import argparse
import numpy as np
import mnist_io
import owl
import owl.elewise as ele
import owl.conv as conv

lazy_cycle = 4

class MNISTCNNModel:
    def __init__(self):
        self.layers = 7 #includeing input layer
        self.filters = [8,16]  
        self.filtersizes = [5,5]
        self.convolution_output_size = 256
        n = pow((((28-self.filtersizes[0] - 1)/2)/2 -self.filtersizes[1] - 1)/2/2, 2)*self.filters[-1]     
        self.convs = [
            conv.Convolver(0, 0, 1, 1),
            conv.Convolver(0, 0, 1, 1),
        ];
        self.poolings = [
            conv.Pooler(2, 2, 2, 2, 0, 0, conv.pool_op.max),
            conv.Pooler(2, 2, 2, 2, 0, 0, conv.pool_op.max)
        ];

    def init_random(self):
        self.weights = [
            owl.randn([self.filtersizes[0], self.filtersizes[0], 1, self.filters[0]], 0.0, 0.1),
            owl.randn([self.filtersizes[1], self.filtersizes[1], self.filters[0], self.filters[1]], 0.0, 0.1),
            owl.randn([128, self.convolution_output_size], 0.0, 0.1),
            owl.randn([10, 128], 0.0, 0.1)
        ];
        self.weightdelta = [
            owl.zeros([self.filtersizes[0], self.filtersizes[0], 1, self.filters[0]]),
            owl.zeros([self.filtersizes[1], self.filtersizes[1], self.filters[0], self.filters[1]]),
            owl.zeros([128, self.convolution_output_size]),
            owl.zeros([10, 128])
        ];
        self.bias = [
            owl.zeros([self.filters[0]]),
            owl.zeros([self.filters[1]]),
            owl.zeros([128, 1]),
            owl.zeros([10, 1])
        ];
        self.biasdelta = [
            owl.zeros([self.filters[0]]),
            owl.zeros([self.filters[1]]),
            owl.zeros([128, 1]),
            owl.zeros([10, 1])
        ];

def print_training_accuracy(o, t, mbsize, prefix):
    predict = o.reshape([10, mbsize]).max_index(0)
    ground_truth = t.reshape([10, mbsize]).max_index(0)
    correct = (predict - ground_truth).count_zero()
    print prefix, 'error: {}'.format((mbsize - correct) * 1.0 / mbsize)

def bpprop(model, samples, label):
    num_layers = model.layers
    num_samples = samples.shape[-1]
    fc_shape = [model.convolution_output_size, num_samples]

    acts = [None] * num_layers
    errs = [None] * num_layers
    weightgrad = [None] * len(model.weights)
    biasgrad = [None] * len(model.bias)

    acts[0] = samples
    acts[1] = ele.relu(model.convs[0].ff(acts[0], model.weights[0], model.bias[0]))
    acts[2] = model.poolings[0].ff(acts[1])
    acts[3] = ele.relu(model.convs[1].ff(acts[2], model.weights[1], model.bias[1]))
    acts[4] = model.poolings[1].ff(acts[3])
    acts[5] = model.weights[2] * acts[4].reshape(fc_shape) + model.bias[2]
    acts[6] = model.weights[3] * acts[5] + model.bias[3]

    out = conv.softmax(acts[6], conv.soft_op.instance)

    errs[6] = out - label
    errs[5] = (model.weights[3].trans() * errs[6]).reshape(acts[5].shape)
    errs[4] = (model.weights[2].trans() * errs[5]).reshape(acts[4].shape)
    errs[3] = ele.relu_back(model.poolings[1].bp(errs[4], acts[4], acts[3]), acts[3])
    errs[2] = model.convs[1].bp(errs[3], acts[2], model.weights[1])
    errs[1] = ele.relu_back(model.poolings[0].bp(errs[2], acts[2], acts[1]), acts[1])
  
    weightgrad[3] = errs[6] * acts[5].trans()
    biasgrad[3] = errs[6].sum(1)  
    weightgrad[2] = errs[5] * acts[4].reshape(fc_shape).trans()
    biasgrad[2] = errs[5].sum(1)
    weightgrad[1] = model.convs[1].weight_grad(errs[3], acts[2], model.weights[1])
    biasgrad[1] = model.convs[1].bias_grad(errs[3])
    weightgrad[0] = model.convs[0].weight_grad(errs[1], acts[0], model.weights[0])
    biasgrad[0] = model.convs[0].bias_grad(errs[1])
    return (out, weightgrad, biasgrad)

def train_network(model, num_epochs=100, minibatch_size=256, lr=0.1, lr_decay= 0.95, mom=0.9, wd=5e-4):
    # load data
    (train_data, test_data) = mnist_io.load_mb_from_mat('mnist_all.mat', minibatch_size / len(devs))
    num_test_samples = test_data[0].shape[0]
    test_samples = owl.from_numpy(test_data[0]).reshape([28, 28, 1, num_test_samples])
    test_labels = owl.from_numpy(test_data[1])
    for i in xrange(num_epochs):
        print "---Epoch #", i
        last = time.time()
        count = 0
        weightgrads = [None] * len(devs)
        biasgrads = [None] * len(devs)
        for (mb_samples, mb_labels) in train_data:
            count += 1
            current_dev = count % len(devs)
            owl.set_device(devs[current_dev])
            num_samples = mb_samples.shape[0]
            data = owl.from_numpy(mb_samples).reshape([28, 28, 1, num_samples])
            label = owl.from_numpy(mb_labels)
            out, weightgrads[current_dev], biasgrads[current_dev] = bpprop(model, data, label)
            if current_dev == 0:
                for k in range(len(model.weights)):
                    model.weightdelta[k] = mom * model.weightdelta[k] - lr / num_samples / len(devs) * multi_dev_merge(weightgrads, 0, k) - lr * wd * model.weights[k]
                    model.biasdelta[k] = mom * model.biasdelta[k] - lr / num_samples / len(devs) * multi_dev_merge(biasgrads, 0, k)
                    model.weights[k] += model.weightdelta[k]
                    model.bias[k] += model.biasdelta[k]
                if count % (len(devs) * lazy_cycle) == 0:
                    print_training_accuracy(out, label, num_samples, 'Training')
                    owl.print_profiler_result()
        print '---End of Epoch #', i, 'time:', time.time() - last
        lr = lr*lr_decay
        # do test
        out, _, _  = bpprop(model, test_samples, test_labels)
        print_training_accuracy(out, test_labels, num_test_samples, 'Testing')

def multi_dev_merge(l, base, layer):
    if len(l) == 1:
        return l[0][layer]
    left = multi_dev_merge(l[:len(l) / 2], base, layer)
    right = multi_dev_merge(l[len(l) / 2:], base + len(l) / 2, layer)
    owl.set_device(base)
    return left + right

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST CNN')
    parser.add_argument('-g', '--gpu', help='use gpus', action='store_true', type=int, default=0)
    parser.add_argument('-m', '--mpi', help='use mpi', action='store_true', type=int, default=0)
    (args, remain) = parser.parse_args()
    #assert(1 <= args.num)
    usempi = False
    if args.mpi == 1:
        usempi = True
    usegpu = False
    if args.gpu == 1:
        usegpu = True
        
    devs = [owl.create_cpu_device()]
    if usempi:
        nodes = owl.get_mpi_node_count()
        if usegpu:
            devs += [owl.create_mpi_device(i,d) for i in range(nodes) for d in owl.get_mpi_device_count(i)]
        else:
            devs += [owl.create_mpi_device(i,0) for i in range(nodes)]
    else:
        if usegpu:
            devs += [owl.create_gpu_device(i) for i in range(owl.get_gpu_device_count())]
        else:
            pass
    owl.set_device(devs[0])
    model = MNISTCNNModel()
    model.init_random()
    train_network(model)
