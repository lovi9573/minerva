import math
import sys
import time
import numpy as np
import owl
from net import Net
import net
from net_helper import CaffeNetBuilder
from caffe import *
from PIL import Image

class NetTrainer:
    ''' Class for training neural network

    Allows user to train using Caffe's network configure format but on multiple GPUs. One
    could use NetTrainer as follows:

        >>> trainer = NetTrainer(solver_file, snapshot, num_gpu)
        >>> trainer.build_net()
        >>> trainer.run()

    :ivar str solver_file: path of the solver file in Caffe's proto format
    :ivar int snapshot: the idx of snapshot to start with
    :ivar int num_gpu: the number of gpu to use
    :ivar int sync_freq: the frequency to stop lazy evaluation and print some information. The frequency means every how many
                         minibatches will the trainer call ``owl.wait_for_all()``. Note that this will influence the training
                         speed. Normally, the higher value is given, the faster the training speed but the more memory is used
                         during execution.
    '''
    def __init__(self, solver_file, snapshot = 0, gpu = 1, sync_freq=1, report=False):
        self.solver_file = solver_file
        self.snapshot = snapshot
        self.num_gpu = gpu
        self.sync_freq = sync_freq
        self.report = report
        if owl.has_mpi():
            self.gpu = []
            if gpu == 1: 
                self.gpu += [owl.create_gpu_device(i) for i in range(owl.get_gpu_device_count())]
                nodes = [owl.get_mpi_device_count(i) for i in range(1,owl.get_mpi_node_count())]
                for n in range(len(nodes)):
                    print "using {} gpu's on node {}\n".format(nodes[n],n+1)
                    self.gpu +=  [owl.create_mpi_device(n+1,i+1) for i in range(nodes[n])]
                self.num_gpu = len(self.gpu)
            else:
        		self.gpu += [owl.create_cpu_device()]
                        self.gpu += [owl.create_mpi_device(n,0) for n in range(1,owl.get_mpi_node_count())]
        		self.num_gpu = len(self.gpu)
		        print "using {} cpu's over all nodes".format(self.num_gpu)
        else:
            if gpu == 1:
                self.gpu = [owl.create_gpu_device(i) for i in range(self.num_gpu)]
                self.num_gpu = len(self.gpu)
                print "using {} gpu devices".format(len(self.gpu))
            else:
                self.gpu = [owl.create_cpu_device()]
                self.num_gpu = len(self.gpu)
                print "using {} cpus".format(len(self.gpu))

    def build_net(self):
        ''' Build network structure using Caffe's proto definition. It will also initialize
        the network either from given snapshot or from scratch (using proper initializer). 
        During initialization, it will first try to load weight from snapshot. If failed, it
        will then initialize the weight accordingly.
        '''
        self.owl_net = Net()
        self.builder = CaffeNetBuilder(self.solver_file)
        self.snapshot_dir = self.builder.snapshot_dir
        print "num_gpu",self.num_gpu
        self.builder.build_net(self.owl_net, self.num_gpu)
        self.owl_net.compute_size()
        self.builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)
	print "net built"

    def run(s):
        ''' Run the training algorithm on multiple GPUs

        The basic logic is similar to the traditional single GPU training code as follows (pseudo-code)::

            for epoch in range(MAX_EPOCH):
                for i in range(NUM_MINI_BATCHES):
                    # load i^th minibatch
                    minibatch = loader.load(i, MINI_BATCH_SIZE)
                    net.ff(minibatch.data)
                    net.bp(minibatch.label)
                    grad = net.gradient()
                    net.update(grad, MINI_BATCH_SIZE)

        With Minerva's lazy evaluation and dataflow engine, we are able to modify the above logic
        to perform data parallelism on multiple GPUs (pseudo-code)::

            for epoch in range(MAX_EPOCH):
                for i in range(0, NUM_MINI_BATCHES, NUM_GPU):
                    gpu_grad = [None for i in range(NUM_GPU)]
                    for gpuid in range(NUM_GPU):
                        # specify which gpu following codes are running on
                        owl.set_device(gpuid)
                        # each minibatch is split among GPUs
                        minibatch = loader.load(i + gpuid, MINI_BATCH_SIZE / NUM_GPU)
                        net.ff(minibatch.data)
                        net.bp(minibatch.label)
                        gpu_grad[gpuid] = net.gradient()
                    net.accumulate_and_update(gpu_grad, MINI_BATCH_SIZE)

        So each GPU will take charge of one *mini-mini batch* training, and since all their ``ff``, ``bp`` and ``gradient``
        calculations are independent among each others, they could be paralleled naturally using Minerva's DAG engine.

        The only problem let is ``accumulate_and_update`` of the the gradient from all GPUs. If we do it on one GPU,
        that GPU would become a bottleneck. The solution is to also partition the workload to different GPUs (pseudo-code)::

            def accumulate_and_update(gpu_grad, MINI_BATCH_SIZE):
                num_layers = len(gpu_grad[0])
                for layer in range(num_layers):
                    upd_gpu = layer * NUM_GPU / num_layers
                    # specify which gpu to update the layer
                    owl.set_device(upd_gpu)
                    for gid in range(NUM_GPU):
                        if gid != upd_gpu:
                            gpu_grad[upd_gpu][layer] += gpu_grad[gid][layer]
                    net.update_layer(layer, gpu_grad[upd_gpu][layer], MINI_BATCH_SIZE)

        Since the update of each layer is independent among each others, the update could be paralleled affluently. Minerva's
        dataflow engine transparently handles the dependency resolving, scheduling and memory copying among different devices,
        so users don't need to care about that.
        '''       
        wgrad = [[] for i in range(s.num_gpu)]
        bgrad = [[] for i in range(s.num_gpu)]
        last = time.time()
        wunits = s.owl_net.get_weighted_unit_ids()
        last_start = time.time()

        start_idx = s.snapshot * s.owl_net.solver.snapshot
        end_idx = s.owl_net.solver.max_iter

        for iteridx in range(start_idx, end_idx):
            # get the learning rate
            if s.owl_net.solver.lr_policy == "poly":
                s.owl_net.current_lr = s.owl_net.base_lr * pow(1 - float(iteridx) / s.owl_net.solver.max_iter, s.owl_net.solver.power)
            elif s.owl_net.solver.lr_policy == "step":
                s.owl_net.current_lr = s.owl_net.base_lr * pow(s.owl_net.solver.gamma, iteridx / s.owl_net.solver.stepsize)

            # train on multi-gpu
            for gpuid in range(s.num_gpu):
                owl.set_device(s.gpu[gpuid])
                print "running gpu {}".format(s.gpu[gpuid])
                s.owl_net.forward('TRAIN')
                s.owl_net.backward('TRAIN')
                for wid in wunits:
                    wgrad[gpuid].append(s.owl_net.units[wid].weightgrad)
                    bgrad[gpuid].append(s.owl_net.units[wid].biasgrad)

            # weight update
            for i in range(len(wunits)):
                wid = wunits[i]
                upd_gpu = i * s.num_gpu / len(wunits)
                owl.set_device(s.gpu[upd_gpu])
                for gid in range(s.num_gpu):
                    if gid == upd_gpu:
                        continue
                    wgrad[upd_gpu][i] += wgrad[gid][i]
                    bgrad[upd_gpu][i] += bgrad[gid][i]
                s.owl_net.units[wid].weightgrad = wgrad[upd_gpu][i]
                s.owl_net.units[wid].biasgrad = bgrad[upd_gpu][i]
                s.owl_net.update(wid)

            if iteridx % s.sync_freq == 0:
                owl.wait_for_all()
                thistime = time.time() - last
                speed = s.owl_net.batch_size * s.sync_freq / thistime
                print "Finished training %d minibatch of %d (time: %s; speed: %s img/s)" % (iteridx, end_idx, thistime, speed)
                last = time.time()

            wgrad = [[] for i in range(s.num_gpu)] # reset gradients
            bgrad = [[] for i in range(s.num_gpu)]

            # decide whether to display loss
            if (iteridx + 1) % (s.owl_net.solver.display) == 0:
                lossunits = s.owl_net.get_loss_units()
                for lu in lossunits:
                    print "Training Loss %s: %f" % (lu.name, lu.getloss())
                print owl.print_profiler_result()

            # decide whether to test
            if (iteridx + 1) % (s.owl_net.solver.test_interval) == 0:
                acc_num = 0
                test_num = 0
                for testiteridx in range(s.owl_net.solver.test_iter[0]):
                    s.owl_net.forward('TEST')
                    all_accunits = s.owl_net.get_accuracy_units()
                    accunit = all_accunits[len(all_accunits)-1]
                    #accunit = all_accunits[0]
                    test_num += accunit.batch_size
                    acc_num += (accunit.batch_size * accunit.acc)
                    print "Accuracy the %d mb: %f" % (testiteridx, accunit.acc)
                    sys.stdout.flush()
                print "Testing Accuracy: %f" % (float(acc_num)/test_num)

            # decide whether to save model
            if (iteridx + 1) % (s.owl_net.solver.snapshot) == 0:
                print "Save to snapshot %d, current lr %f" % ((iteridx + 1) / (s.owl_net.solver.snapshot), s.owl_net.current_lr)
                s.builder.save_net_to_file(s.owl_net, s.snapshot_dir, (iteridx + 1) / (s.owl_net.solver.snapshot))
            sys.stdout.flush()
            
            #print stats
            if s.report:
                owl.print_profiler_result()
                

    def gradient_checker(s, checklayer_name):
        ''' Check backpropagation on multiple GPUs
        '''
        h = 1e-2
        threshold = 1e-4
        checklayer = s.owl_net.units[s.owl_net.name_to_uid[checklayer_name][0]] 
        
        losslayer = []
        for i in xrange(len(s.owl_net.units)):
            if isinstance(s.owl_net.units[i], net.SoftmaxUnit):
                losslayer.append(i)
       
        last = None
        '''
        wunits = []
        for i in xrange(len(s.owl_net.units)):
            if isinstance(s.owl_net.units[i], net.WeightedComputeUnit):
                wunits.append(i)
        '''
        wunits = s.owl_net.get_weighted_unit_ids()
        accunits = s.owl_net.get_accuracy_units()
        owl.set_device(s.gpu[0])
        
        for iteridx in range(100):
            #disturb the weights
            oriweight = checklayer.weight
            npweight = checklayer.weight.to_numpy()
            weightshape = np.shape(npweight)
            npweight = npweight.reshape(np.prod(weightshape[0:len(weightshape)]))
            position = np.random.randint(0, np.shape(npweight)[0])
            disturb = np.zeros(np.shape(npweight), dtype = np.float32)
            disturb[position] = h
            oriposval = npweight[position]
            npweight += disturb
            newposval = npweight[position]
            npweight = npweight.reshape(weightshape)
            checklayer.weight = owl.from_numpy(npweight)

            all_loss = 0
            # train on multi-gpu

            s.owl_net.forward_check()
            for i in range(len(losslayer)):
                if len(s.owl_net.units[losslayer[i]].loss_weight) == 1:
                    all_loss += (s.owl_net.units[losslayer[i]].getloss() * s.owl_net.units[losslayer[i]].loss_weight[0])
                else:
                    all_loss += s.owl_net.units[losslayer[i]].getloss()

            #get origin loss
            checklayer.weight = oriweight
            ori_all_loss = 0
            # train on multi-gpu
            s.owl_net.forward_check()
            for i in range(len(losslayer)):
                if len(s.owl_net.units[losslayer[i]].loss_weight) == 1:
                    ori_all_loss += (s.owl_net.units[losslayer[i]].getloss() * s.owl_net.units[losslayer[i]].loss_weight[0])
                else:
                    ori_all_loss += s.owl_net.units[losslayer[i]].getloss()

            s.owl_net.backward('TEST')
            #get analytic gradient
            npgrad = checklayer.weightgrad.to_numpy()
            npgrad = npgrad.reshape(np.prod(weightshape[0:len(weightshape)]))
            analy_grad = npgrad[position] /  s.owl_net.units[losslayer[i]].out.shape[1]
           
            num_grad = (all_loss - ori_all_loss) / h
            
            info = "Gradient Check at positon: %d analy: %f num: %f ratio: %f" % (position, analy_grad, num_grad, analy_grad / num_grad)
            print info

class NetTester:
    ''' Class for performing testing, it can be single-view or multi-view, can be top-1 or top-5

    Run it as::
        >>> tester = NetTester(solver_file, softmax_layer, accuracy_layer, snapshot, gpu_idx)
        >>> tester.build_net()
        >>> tester.run(multiview)

    :ivar str solver_file: path of the solver file in Caffe's proto format
    :ivar int snapshot: the snapshot for testing
    :ivar str softmax_layer_name: name of the softmax layer that produce prediction 
    :ivar str accuracy_layer_name: name of the accuracy layer that produce prediction 
    :ivar int gpu_idx: which gpu to perform the test
    :ivar bool multiview: whether to use multiview tester
    '''
    def __init__(self, solver_file, softmax_layer_name, accuracy_layer_name, snapshot, gpu_idx = 0):
        self.solver_file = solver_file
        self.softmax_layer_name = softmax_layer_name
        self.accuracy_layer_name = accuracy_layer_name
        self.snapshot = snapshot
        self.gpu = owl.create_gpu_device(gpu_idx)
        owl.set_device(self.gpu)

    def build_net(self):
        self.owl_net = Net()
        self.builder = CaffeNetBuilder(self.solver_file)
        self.snapshot_dir = self.builder.snapshot_dir
        self.builder.build_net(self.owl_net)
        self.owl_net.compute_size('TEST')
        self.builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)

    def run(s, multiview):
        #multi-view test
        acc_num = 0
        test_num = 0
        loss_unit = s.owl_net.units[s.owl_net.name_to_uid[s.softmax_layer_name][0]] 
        accunit = s.owl_net.units[s.owl_net.name_to_uid[s.accuracy_layer_name][0]] 
        data_unit = None
        for data_idx in range(len(s.owl_net.data_layers)):
            for i in range(len(s.owl_net.name_to_uid[s.owl_net.data_layers[data_idx]])):
                if s.owl_net.units[s.owl_net.name_to_uid[s.owl_net.data_layers[data_idx]][i]].params.include[0].phase == 1:
                    data_unit = s.owl_net.units[s.owl_net.name_to_uid[s.owl_net.data_layers[data_idx]][i]]
        assert(data_unit)
        if multiview == True:
            data_unit.multiview = True

        for testiteridx in range(s.owl_net.solver.test_iter[0]):
            if multiview == True:
                for i in range(10): 
                    s.owl_net.forward('TEST')
                    if i == 0:
                        softmax_val = loss_unit.ff_y
                        batch_size = softmax_val.shape[1]
                        softmax_label = loss_unit.y
                    else:
                        softmax_val = softmax_val + loss_unit.ff_y
                test_num += batch_size
                if accunit.top_k == 5:
                    predict = softmax_val.to_numpy()
                    top_5 = np.argsort(predict, axis=1)[:,::-1]
                    ground_truth = softmax_label.max_index(0).to_numpy()
                    correct = 0
                    for i in range(batch_size):
                        for t in range(5):
                            if ground_truth[i] == top_5[i,t]:
                                correct += 1
                                break
                    acc_num += correct
                else:
                    predict = softmax_val.max_index(0)
                    truth = softmax_label.max_index(0)
                    correct = (predict - truth).count_zero()
                    acc_num += correct
            else:
                s.owl_net.forward('TEST')
                all_accunits = s.owl_net.get_accuracy_units()
                batch_size = accunit.batch_size
                test_num += batch_size
                acc_num += (batch_size * accunit.acc)
                correct = batch_size * accunit.acc
            print "Accuracy of the %d mb: %f, batch_size: %d, current mean accuracy: %f" % (testiteridx, (correct * 1.0)/batch_size, batch_size, float(acc_num)/test_num)
            sys.stdout.flush()
        print "Testing Accuracy: %f" % (float(acc_num)/test_num)

class FeatureExtractor:
    ''' Class for extracting trained features
    Feature will be stored in a txt file as a matrix. The size of the feature matrix is [num_img, feature_dimension]

    Run it as::
        >>> extractor = FeatureExtractor(solver_file, snapshot, gpu_idx)
        >>> extractor.build_net()
        >>> extractor.run(layer_name, feature_path)

    :ivar str solver_file: path of the solver file in Caffe's proto format
    :ivar int snapshot: the snapshot for testing
    :ivar str layer_name: name of the ayer that produce feature 
    :ivar int gpu_idx: which gpu to perform the test
    '''
    def __init__(self, solver_file, snapshot, gpu_idx = 0):
        self.solver_file = solver_file
        self.snapshot = snapshot
        self.gpu = owl.create_gpu_device(gpu_idx)
        owl.set_device(self.gpu)

    def build_net(self):
        self.owl_net = Net()
        self.builder = CaffeNetBuilder(self.solver_file)
        self.snapshot_dir = self.builder.snapshot_dir
        self.builder.build_net(self.owl_net)
        self.owl_net.compute_size('TEST')
        self.builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)

    def run(s, layer_name, feature_path):
        ''' Run feature extractor

        :param str layer_name: the layer to extract feature from
        :param str feature_path: feature output path
        '''
        feature_unit = s.owl_net.units[s.owl_net.name_to_uid[layer_name][0]] 
        feature_file = open(feature_path, 'w')
        batch_dir = 0
        for testiteridx in range(s.owl_net.solver.test_iter[0]):
            s.owl_net.forward('TEST')
            feature = feature_unit.out.to_numpy()
            feature_shape = np.shape(feature)
            img_num = feature_shape[0]
            feature_length = np.prod(feature_shape[1:len(feature_shape)])
            feature = np.reshape(feature, [img_num, feature_length])
            for imgidx in range(img_num):
                for feaidx in range(feature_length):
                    info ='%f ' % (feature[imgidx, feaidx])
                    feature_file.write(info)
                feature_file.write('\n')
            print "Finish One Batch %d" % (batch_dir)
            batch_dir += 1
        feature_file.close()

class FilterVisualizer:
    ''' Class of filter visualizer.
    Find the most interested patches of a filter to demostrate the pattern that filter insterested in. It first read in several images to conduct feed-forward and find the patches have the biggest activation value for a filter. Those patches usually contains the pattern of that filter. 

    :ivar str solver_file: name of the solver_file, it will tell Minerva the network configuration and model saving path 
    :ivar snapshot: saved model snapshot index
    :ivar str layer_name: name of the layer that will be viusualized, we will visualize all the filters in that layer in one time
    :ivar str result_path: path for the result of visualization, filtervisualizer will generate a jpg contains the nine selected patches for each filter in layer_name and save the image under result path. 
    :ivar gpu: the gpu to run testing

    '''
    
    
    def __init__(self, solver_file, snapshot, layer_name, result_path, gpu_idx = 0):
        self.solver_file = solver_file
        self.snapshot = snapshot
        self.layer_name = layer_name
        self.result_path = result_path
        self.gpu = owl.create_gpu_device(gpu_idx)
        owl.set_device(self.gpu)

    def build_net(self):
        self.owl_net = Net()
        self.builder = CaffeNetBuilder(self.solver_file)
        self.snapshot_dir = self.builder.snapshot_dir
        self.builder.build_net(self.owl_net)
        self.owl_net.compute_size('TEST')
        self.builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)

    def run(s):
        #Need Attention, here we may have multiple data layer, just choose the TEST layer
        data_unit = None
        for data_idx in range(len(s.owl_net.data_layers)):
            for i in range(len(s.owl_net.name_to_uid[s.owl_net.data_layers[data_idx]])):
                if s.owl_net.units[s.owl_net.name_to_uid[s.owl_net.data_layers[data_idx]][i]].params.include[0].phase == 1:
                    data_unit = s.owl_net.units[s.owl_net.name_to_uid[s.owl_net.data_layers[data_idx]][i]]
        assert(data_unit)
       
        bp = BlobProto()
        #get mean file
        if len(data_unit.params.transform_param.mean_file) == 0:
            mean_data = np.ones([3, 256, 256], dtype=np.float32)
            assert(len(data_unit.params.transform_param.mean_value) == 3)
            mean_data[0] = data_unit.params.transform_param.mean_value[0]
            mean_data[1] = data_unit.params.transform_param.mean_value[1]
            mean_data[2] = data_unit.params.transform_param.mean_value[2]
            h_w = 256
        else:    
            with open(data_unit.params.transform_param.mean_file, 'rb') as f:
                bp.ParseFromString(f.read())
            mean_narray = np.array(bp.data, dtype=np.float32)
            h_w = np.sqrt(np.shape(mean_narray)[0] / 3)
            mean_data = np.array(bp.data, dtype=np.float32).reshape([3, h_w, h_w])
        #get the cropped img
        crop_size = data_unit.params.transform_param.crop_size
        crop_h_w = (h_w - crop_size) / 2
        mean_data = mean_data[:, crop_h_w:crop_h_w + crop_size, crop_h_w:crop_h_w + crop_size]

        feature_unit = s.owl_net.units[s.owl_net.name_to_uid[s.layer_name][0]] 
        batch_dir = 0
        #we use 10000 images to conduct visualization
        all_data = np.zeros([10000, 3, crop_size, crop_size], dtype=np.float32)
        feature_shape = feature_unit.out_shape
        all_feature = np.zeros([10000, feature_shape[2], feature_shape[1], feature_shape[0]], dtype=np.float32) 
        
        print 'Begin Generating Activations from Testing Set'
        curimg = 0
        for testiteridx in range(s.owl_net.solver.test_iter[0]):
            s.owl_net.forward('TEST')
            feature = feature_unit.out.to_numpy()
            batch_size = np.shape(feature)[0]
            all_feature[curimg:curimg+batch_size,:] = feature
            data = data_unit.out.to_numpy()
            all_data[curimg:curimg+batch_size,:] = data
            curimg += batch_size
            #HACK TODO: only take 10000 images
            if curimg >= 10000:
                break
            info = 'Now Processed %d images' % (curimg)
            print info
        print 'Begin Selecting Patches'
        #get the result 
        patch_shape = feature_unit.rec_on_ori
        min_val = -float('inf') 
        
        #add back the mean file
        for i in range(np.shape(all_data)[0]):
            all_data[i,:,:,:] += mean_data
       
        if len(feature_shape) == 4:
            #iter for each filter, for each filter, we choose nine patch from different image
            for i in range(feature_shape[2]):
                #create the result image for nine patches
                res_img = np.zeros([feature_unit.rec_on_ori * 3, feature_unit.rec_on_ori * 3, 3])
                filter_feature = np.copy(all_feature[:,i,:,:])
                for patchidx in range(9):
                    maxidx = np.argmax(filter_feature)
                    colidx = maxidx % feature_shape[0]
                    maxidx = (maxidx - colidx) / feature_shape[0]
                    rowidx = maxidx % feature_shape[1]
                    maxidx = (maxidx - rowidx) / feature_shape[1]
                    imgidx = maxidx
                    info = '%d %d %d' % (imgidx, rowidx, colidx)
                    filter_feature[imgidx,:,:] = min_val
                    
                    #get the patch place
                    patch_start_row = max(0,feature_unit.start_on_ori + rowidx * feature_unit.stride_on_ori)
                    patch_end_row = min(feature_unit.start_on_ori + rowidx * feature_unit.stride_on_ori + feature_unit.rec_on_ori, data_unit.crop_size)
                    if patch_start_row == 0:
                        patch_end_row = feature_unit.rec_on_ori
                    if patch_end_row == data_unit.crop_size:
                        patch_start_row = data_unit.crop_size - feature_unit.rec_on_ori

                    patch_start_col = max(0,feature_unit.start_on_ori + colidx * feature_unit.stride_on_ori)
                    patch_end_col = min(feature_unit.start_on_ori + colidx * feature_unit.stride_on_ori + feature_unit.rec_on_ori, data_unit.crop_size)
                    if patch_start_col == 0:
                        patch_end_col = feature_unit.rec_on_ori
                    if patch_end_col == data_unit.crop_size:
                        patch_start_col = data_unit.crop_size - feature_unit.rec_on_ori
                    
                    patch = all_data[imgidx, :, patch_start_row:patch_end_row, patch_start_col:patch_end_col]

                    #save img to image
                    row_in_res = patchidx / 3
                    col_in_res = patchidx % 3
                    st_row = row_in_res * patch_shape 
                    st_col = col_in_res * patch_shape
                    #turn gbr into rgb
                    res_img[st_row:st_row+patch_end_row - patch_start_row, st_col:st_col + patch_end_col - patch_start_col, 2] = patch[0,:,:]
                    res_img[st_row:st_row+patch_end_row - patch_start_row, st_col:st_col + patch_end_col - patch_start_col, 1] = patch[1,:,:]
                    res_img[st_row:st_row+patch_end_row - patch_start_row, st_col:st_col + patch_end_col - patch_start_col, 0] = patch[2,:,:]

                #save img
                res_img = Image.fromarray(res_img.astype(np.uint8))
                res_path = '%s/%d.jpg' % (s.result_path, i)
                print res_path
                res_img.save(res_path, format = 'JPEG')
        else:
            #Fully Layers
            #iter for each filter, for each filter, we choose nine patch from different image
            print feature_shape
            for i in range(feature_shape[0]):
                #create the result image for nine patches
                res_img = np.zeros([data_unit.crop_size * 3, data_unit.crop_size * 3, 3])
                filter_feature = np.copy(all_feature[:,i])
                for patchidx in range(9):
                    maxidx = np.max_index(filter_feature)
                    imgidx = maxidx
                    filter_feature[imgidx] = min_val

                    #save img to image
                    row_in_res = patchidx / 3
                    col_in_res = patchidx % 3
                    st_row = row_in_res * data_unit.crop_size 
                    st_col = col_in_res * data_unit.crop_size
                    #turn gbr into rgb
                    patch = all_data[imgidx,:,:,:]
                    res_img[st_row:st_row+data_unit.crop_size,st_col:st_col+data_unit.crop_size, 2] = patch[0,:,:]
                    res_img[st_row:st_row+data_unit.crop_size,st_col:st_col+data_unit.crop_size, 1] = patch[1,:,:]
                    res_img[st_row:st_row+data_unit.crop_size,st_col:st_col+data_unit.crop_size, 0] = patch[2,:,:]
                #save img
                res_img = Image.fromarray(res_img.astype(np.uint8))
                res_path = '%s/%d.jpg' % (s.result_path, i)
                print res_path
                res_img.save(res_path, format = 'JPEG')
