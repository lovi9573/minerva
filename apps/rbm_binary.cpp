/*
 * rbm_binary.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: jlovitt
 */

#include <cstdio>
#include <minerva.h>
#include <iomanip>
#include <fstream>
#include "mnist_raw.h"

using namespace minerva;


#define DATA_MIN 0
#define DATA_MAX 255

#define N_HIDDEN 128
#define N_EPOCHS 10
#define BATCH_SIZE 64
#define MOMENTUM 0.9





int main(int argc, char** argv){

	//Initialize minerva
	printf("mineva init\n");
	IMinervaSystem::Init(&argc,&argv);
	auto&& mi = IMinervaSystem::Interface();
	uint64_t cpu = mi.CreateCpuDevice();
	mi.SetDevice(cpu);


	//Load the training data
	printf("load data\n");
	int n_samples,sample_size;
	MnistData dp(argv[1]);
	n_samples = dp.nSamples();
	sample_size = dp.SampleSize();
	printf("sample size %d",sample_size);

	//Initialize arrays
	printf("Initialize data structures\n");
	NArray weights = NArray::Randn({ sample_size, N_HIDDEN},0,1);
	NArray bias_v = NArray::Zeros({1,sample_size});
	NArray bias_h = NArray::Zeros({1,N_HIDDEN});

	NArray d_weights = NArray::Zeros({sample_size, N_HIDDEN});
	NArray d_bias_v = NArray::Zeros({1,sample_size});
	NArray d_bias_h = NArray::Zeros({1,N_HIDDEN});

	int n_batches = n_samples/BATCH_SIZE;

	//Begin training
	for(int i_epoch = 0; i_epoch< N_EPOCHS; i_epoch++){
		printf("Epoch %d\n",i_epoch);
		float mse = 0.0;
		for(int i_batch = 0; i_batch < n_batches; i_batch++){
			printf("\t Batch %d/%d\n",i_batch,n_batches);
			//Get minibatch
			shared_ptr<float> batch = dp.GetNextBatch(BATCH_SIZE);
			NArray visible = NArray::MakeNArray({BATCH_SIZE,sample_size}, batch);

			//Apply momentum
			//printf("momentum\n");
			d_weights *= MOMENTUM;
			d_bias_v *= MOMENTUM;
			d_bias_h *= MOMENTUM;

			//Positive Phase
			//printf("positive phase\n");
			NArray in_h = visible*weights + bias_h;
			NArray hidden = 1.0/(1.0 + Elewise::Exp(-in_h));

			d_weights += visible.Trans()*hidden;
			d_bias_v += visible.Sum(0);
			d_bias_h += hidden.Sum(0);

			//Sample Hiddens

			//Negative Phase

			//Update Weights

			//Compute Error

		}//End batches

	}//End epochs

	return 0;

}

