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
#include <string>
#include "mnist_raw.h"


using namespace minerva;

//#define DIAGNOSTIC

#define DATA_MIN 0
#define DATA_MAX 255

#define N_HIDDEN 10
#define N_EPOCHS 10
#define BATCH_SIZE 60
#define MOMENTUM 0.9
#define LR 0.01





int main(int argc, char** argv){

	FileFormat ff;
	ff.binary = false;

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
	printf("%d samples of size %d\n",n_samples,sample_size);

	//Initialize arrays
	printf("Initialize data structures\n");
	NArray weights = NArray::Randn({sample_size, N_HIDDEN},0,1);
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
			if(i_batch%100 == 0){
				printf("\t Batch %d/%d\n",i_batch,n_batches);
			}
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
			NArray uniform_randoms = NArray::RandUniform(hidden.Size(), 1.0);
			NArray sampled_hiddens = hidden > uniform_randoms;

			//Negative Phase
			NArray in_v = sampled_hiddens*weights.Trans() + bias_v;
			NArray reconstruction = 1.0/(1.0 + Elewise::Exp(-in_v));
			in_h = reconstruction*weights + bias_h;
			hidden = 1.0/(1.0 + Elewise::Exp(-in_h));

			d_weights -= reconstruction.Trans()*hidden;
			d_bias_v -= reconstruction.Sum(0);
			d_bias_h -= hidden.Sum(0);

			//d_weights.ToStream(std::cout,ff);
			//std::cout <<"\n";
			//Update Weights
			d_weights = d_weights * LR/BATCH_SIZE ;
			bias_v  = d_bias_v * LR/BATCH_SIZE ;
			bias_h  = d_bias_h * LR/BATCH_SIZE ;

			//Look for update histogram problems
			NArray weight_hist = d_weights.Histogram(10);
			NArray bias_v_hist = bias_v.Histogram(10);
			NArray bias_h_hist = bias_h.Histogram(10);
#ifdef DIAGNOSTIC
			if(weight_hist.Get().get()[0] < -100 || weight_hist.Get().get()[7] > 100){
				std::cout << "Weight Deltas:\n";
				weight_hist.ToStream(std::cout,ff);
			}
			if(bias_v.Get().get()[0] < -100 || bias_v.Get().get()[7] > 100){
				std::cout << "Visible Bias Deltas:\n";
				bias_v.ToStream(std::cout,ff);
			}
			if(bias_h.Get().get()[0] < -100 || bias_h.Get().get()[7] > 100){
				std::cout << "Weight Deltas:\n";
				bias_h.ToStream(std::cout,ff);
			}

			//Output weights as a set of images
			if(i_batch%100 == 0){
				std::cout << "Weight Deltas:\n";
				weight_hist.ToStream(std::cout,ff);
				string fname = "training_weights_"+std::to_string(i_epoch)+"_"+std::to_string(i_batch)+".jpg";
			}
#endif

			weights += d_weights;
			bias_v  += d_bias_v ;
			bias_h  += d_bias_h ;

			//Compute Error
			NArray diff = reconstruction - visible;
			NArray sqrdiff = Elewise::Mult(diff,diff);
			float sum = sqrdiff.Sum();
			float error = sum/sqrdiff.Size().Prod();
			mse += error;
		}//End batches
		mse = mse/n_batches;
		printf("MSE: %f\n",mse);

	}//End epochs
	ofstream of;
	of.open(argv[2],std::ifstream::out);
	weights.ToStream(of,ff);
	of.close();

	return 0;

}

