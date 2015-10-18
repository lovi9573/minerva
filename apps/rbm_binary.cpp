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

#define DIAGNOSTIC

#define DATA_MIN 0
#define DATA_MAX 255

#define N_HIDDEN 15
#define N_EPOCHS 10
#define BATCH_SIZE 64
#define MOMENTUM 0.5
#define LR 0.01

int main(int argc, char** argv) {
	if (argc != 3) {
		printf(
				"Use: rbm_binary <path to input data> <weight output filename>\n\tWeights are saved such that hidden layer weights are consecutive and visible weights interleaved.\n");
		exit(0);
	}

	FileFormat ff;
	ff.binary = false;

	//Initialize minerva
	printf("minerva init\n");
	IMinervaSystem::Init(&argc, &argv);
	auto&& mi = IMinervaSystem::Interface();
	uint64_t gpu = -1;
	bool has_gpu = false;
	if (mi.device_manager().GetGpuDeviceCount() > 0) {
		gpu = mi.CreateGpuDevice(0);
		has_gpu = true;
	}
	uint64_t cpu = mi.CreateCpuDevice();
	mi.SetDevice(cpu);

	//Load the training data
	printf("load data\n");
	int n_samples, sample_size;
	MnistData dp(argv[1]);
	n_samples = dp.nSamples();
	sample_size = dp.SampleSize();
	printf("\t%d samples of size %d\n", n_samples, sample_size);

	//Initialize arrays
	printf("Initialize data structures\n");
	NArray weights = NArray::Randn( { N_HIDDEN, sample_size }, 0, 1);  //H x V
	NArray bias_v = NArray::Zeros( { sample_size, 1 });
	NArray bias_h = NArray::Zeros( { N_HIDDEN, 1 });

	NArray d_weights = NArray::Zeros( { N_HIDDEN, sample_size, });
	NArray d_bias_v = NArray::Zeros( { sample_size, 1 });
	NArray d_bias_h = NArray::Zeros( { N_HIDDEN, 1 });
	NArray sqrdiff, visible, reconstruction;

	int n_batches = n_samples / BATCH_SIZE;

	//Begin training
	for (int i_epoch = 0; i_epoch < N_EPOCHS; i_epoch++) {
		printf("Epoch %d\n", i_epoch);
		float mse = 0.0;
		for (int i_batch = 0; i_batch < n_batches; i_batch++) {
			if (has_gpu) {
				mi.SetDevice(gpu);
			}
			if (i_batch % 100 == 0) {
				printf("\t Batch %d/%d\n", i_batch, n_batches);
			}
			//Get minibatch
			shared_ptr<float> batch = dp.GetNextBatch(BATCH_SIZE);
			visible = NArray::MakeNArray( { sample_size, BATCH_SIZE }, batch); //V x B

			//Apply momentum
			d_weights *= MOMENTUM;
			d_bias_v *= MOMENTUM;
			d_bias_h *= MOMENTUM;

			//Positive Phase
			NArray in_h = weights * visible + bias_h;
			NArray hidden = 1.0 / (1.0 + Elewise::Exp(-in_h)); // H x B
			d_weights += hidden * visible.Trans();
			d_bias_v += visible.Sum(1);
			d_bias_h += hidden.Sum(1);

			//Sample Hiddens
			mi.SetDevice(cpu);
			NArray uniform_randoms = NArray::RandUniform(hidden.Size(), 1.0);
			if (has_gpu) {
				mi.SetDevice(gpu);
			}
			NArray sampled_hiddens = hidden > uniform_randoms; //H x B

			//Negative Phase
			NArray in_v = weights.Trans() * sampled_hiddens + bias_v;
			reconstruction = 1.0 / (1.0 + Elewise::Exp(-in_v)); //V x B
			in_h = weights * reconstruction + bias_h;
			hidden = 1.0 / (1.0 + Elewise::Exp(-in_h));  //H x B

			d_weights -= hidden * reconstruction.Trans();
			d_bias_v -= reconstruction.Sum(1);
			d_bias_h -= hidden.Sum(1);

			//d_weights.ToStream(std::cout,ff);
			//std::cout <<"\n";
			//Update Weights
			d_weights = d_weights * LR / BATCH_SIZE;
			d_bias_v = d_bias_v * LR / BATCH_SIZE;
			d_bias_h = d_bias_h * LR / BATCH_SIZE;

			weights += d_weights;
			bias_v += d_bias_v;
			bias_h += d_bias_h;
#ifdef DIAGNOSTIC
			//Look for update histogram problems
			mi.SetDevice(cpu);
			NArray weight_hist = d_weights.Histogram(10);
			NArray bias_v_hist = d_bias_v.Histogram(10);
			NArray bias_h_hist = d_bias_h.Histogram(10);
			if(weight_hist.Get().get()[0] < -50 || weight_hist.Get().get()[7] > 50) {
				std::cout << "Weight Deltas:\n";
				weight_hist.ToStream(std::cout,ff);
			}
			if(bias_v.Get().get()[0] < -50 || bias_v.Get().get()[7] > 50) {
				std::cout << "Visible Bias Deltas:\n";
				bias_v_hist.ToStream(std::cout,ff);
			}
			if(bias_h.Get().get()[0] < -50 || bias_h.Get().get()[7] > 50) {
				std::cout << "Weight Deltas:\n";
				bias_h_hist.ToStream(std::cout,ff);
			}
			if(has_gpu) {
				mi.SetDevice(gpu);
			}

			//Compute Error
			NArray diff = reconstruction - visible;
			sqrdiff = Elewise::Mult(diff,diff);
			NArray sum0 = sqrdiff.Sum(0).Sum(0);
			mi.SetDevice(cpu);
			float error = sum0.Sum() / sqrdiff.Size().Prod();
			mse += error;
#endif
			if (i_batch % 3000 == 0) {
				//mi.WaitForAll();
			}

		}			//End batches
#ifdef DIAGNOSTIC
		if(has_gpu) {
			mi.SetDevice(gpu);
		}
		mse = mse / n_batches;
		printf("MSE: %f\n", mse);
		//write an error side by side img
		NArray vis = Slice(visible,1,0,1);
		NArray rec = Slice(reconstruction,1,0,1);
		NArray sdif = Slice(sqrdiff,1,0,1);
		NArray error_side_by_side = Concat( {vis.Trans(),rec.Trans(),sdif.Trans()},0);
		printf("lkdsj\n");
		ofstream errof;
		errof.open("error_img"+std::to_string(i_epoch), std::ifstream::out);
		error_side_by_side.ToStream(errof,ff);
		errof.close();

		//write the current weights
		ofstream wof;
		wof.open(argv[2]+std::to_string(i_epoch), std::ifstream::out);
		weights.ToStream(wof, ff);
		wof.close();
#endif
	}			//End epochs
	mi.PrintProfilerResults();
	ofstream of;
	of.open(argv[2], std::ifstream::out);
	weights.ToStream(of, ff);
	of.close();

	return 0;

}

