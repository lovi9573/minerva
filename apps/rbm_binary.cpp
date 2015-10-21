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


#define N_HIDDEN 64
#define N_EPOCHS 10
#define BATCH_SIZE 10
#define MOMENTUM 0.0
#define LR 0.1
#define GIBBS_SAMPLING_STEPS 1
#define SYNCHRONIZATION_RATE 200



void writeNArray(NArray& array, std::string filename){
	FileFormat ff;
	ff.binary = false;
	ofstream of;
	of.open(filename, std::ifstream::out);
	Scale tscale = array.Size();
	int tx = (int)sqrt(tscale[1]);
	int ty = tscale[1]/tx;
	of  << tx << " " << ty << " " << tscale[1] << "\n";
	array.ToStream(of, ff);
	of.close();
}



int main(int argc, char** argv) {
	if (argc != 3) {
		printf(
				"Use: rbm_binary <path to input data> <weight output filename>\n\tWeights are saved such that hidden layer weights are consecutive and visible weights interleaved.\n");
		exit(0);
	}

	FileFormat ff;
	ff.binary = false;
	std::string output_base(argv[2]);
	bool persistent = false;

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
	NArray weights = NArray::Randn( { N_HIDDEN, sample_size }, 0, .2);  //H x V
	NArray bias_v = NArray::Zeros( { sample_size, 1 });
	NArray bias_h = NArray::Zeros( { N_HIDDEN, 1 });

	//write the current weights
	ofstream twof;
	twof.open(output_base +"_weights_e_init", std::ifstream::out);
	Scale tscale = weights.Size();
	int tx = (int)sqrt(tscale[1]);
	int ty = tscale[1]/tx;
	twof  << tx << " " << ty << " " << tscale[0] << "\n";
	weights.Trans().ToStream(twof, ff);
	twof.close();


	NArray d_weights = NArray::Zeros( { N_HIDDEN, sample_size, });
	NArray d_bias_v = NArray::Zeros( { sample_size, 1 });
	NArray d_bias_h = NArray::Zeros( { N_HIDDEN, 1 });

	NArray d_weights_ave = NArray::Zeros( { N_HIDDEN, sample_size, });
	NArray d_bias_v_ave = NArray::Zeros( { sample_size, 1 });
	NArray d_bias_h_ave = NArray::Zeros( { N_HIDDEN, 1 });
	NArray sqrdiff, visible, reconstruction, hidden, chain_visible;
	bool is_chain_init = false;

	int n_batches = n_samples / BATCH_SIZE;

	//Begin training
	for (int i_epoch = 0; i_epoch < N_EPOCHS; i_epoch++) {
		printf("Epoch %d\n", i_epoch);
#ifdef DIAGNOSTIC
		float mse = 0.0;
		d_weights_ave = NArray::Zeros( { N_HIDDEN, sample_size, });
		d_bias_v_ave = NArray::Zeros( { sample_size, 1 });
		d_bias_h_ave = NArray::Zeros( { N_HIDDEN, 1 });
#endif
		for (int i_batch = 0; i_batch < n_batches; i_batch++) {
			if (has_gpu) {
				mi.SetDevice(gpu);
			}
			if (i_batch % SYNCHRONIZATION_RATE == 0) {
				printf("\t Batch %d/%d\n", i_batch, n_batches);
			}
			//Get minibatch
			shared_ptr<float> batch = dp.GetNextBatch(BATCH_SIZE);
			visible = NArray::MakeNArray( { sample_size, BATCH_SIZE }, batch); //V x B
			if(persistent && !is_chain_init){
				chain_visible = visible;
				is_chain_init = true;
			}

			//Apply momentum
			d_weights *= MOMENTUM;
			d_bias_v *= MOMENTUM;
			d_bias_h *= MOMENTUM;

			//Positive Phase
			NArray in_h = weights * visible + bias_h;
			hidden = 1.0 / (1.0 + Elewise::Exp(-in_h)); // H x B
			NArray d_weights_p = hidden * visible.Trans();
			NArray d_bias_v_p = visible.Sum(1);
			NArray d_bias_h_p = hidden.Sum(1);

			if(persistent){
				NArray in_h = weights * chain_visible + bias_h;
				hidden = 1.0 / (1.0 + Elewise::Exp(-in_h)); // H x B
			}

			for(int gibbs_step = 0; gibbs_step < GIBBS_SAMPLING_STEPS; gibbs_step++){
				//Sample Hiddens
				mi.SetDevice(cpu);
				NArray uniform_randoms = NArray::RandUniform(hidden.Size(), 1.0);
				if (has_gpu) {
					mi.SetDevice(gpu);
				}
				NArray sampled_hiddens = hidden > uniform_randoms; //H x B

				NArray in_v = weights.Trans() * sampled_hiddens + bias_v;
				reconstruction = 1.0 / (1.0 + Elewise::Exp(-in_v)); //V x B
				in_h = weights * reconstruction + bias_h;
				hidden = 1.0 / (1.0 + Elewise::Exp(-in_h));  //H x B
			}
			if(persistent){
				chain_visible = reconstruction;
			}

			//write the current reconstruction
			ofstream rof;
			rof.open(output_base +"_recon_e"+std::to_string(i_epoch)+"_b"+std::to_string(i_batch), std::ifstream::out);
			Scale rscale = reconstruction.Size();
			int rx = (int)sqrt(rscale[0]);
			int ry = rscale[0]/rx;
			rof  << rx << " " << ry << " " << rscale[1] << "\n";
			reconstruction.ToStream(rof,ff);
			rof.close();

			//Negative Phase
			NArray d_weights_n = hidden * reconstruction.Trans();
			NArray d_bias_v_n = reconstruction.Sum(1);
			NArray d_bias_h_n = hidden.Sum(1);

			//write the current positive and negative weight update
			ofstream wof;
			wof.open(output_base +"_weight_update_e"+std::to_string(i_epoch)+"_b"+std::to_string(i_batch), std::ifstream::out);
			Scale scale = weights.Size();
			int x = (int)sqrt(scale[1]);
			int y = scale[1]/x;
			wof  << x << " " << y << " " << (scale[0]*2) << "\n";
			(d_weights_p* LR / BATCH_SIZE).Trans().ToStream(wof,ff);
			(d_weights_n* LR / BATCH_SIZE).Trans().ToStream(wof, ff);
			wof.close();

			//d_weights.ToStream(std::cout,ff);
			//std::cout <<"\n";
			//Update Weights
			d_weights += (d_weights_p - d_weights_n);
			d_bias_v += (d_bias_v_p - d_bias_v_n);
			d_bias_h += (d_bias_h_p - d_bias_h_n);

			weights += d_weights* LR / BATCH_SIZE;
			bias_v += d_bias_v* LR / BATCH_SIZE;
			bias_h += d_bias_h* LR / BATCH_SIZE;

			d_weights_ave += d_weights* LR / BATCH_SIZE;
			d_bias_v_ave += d_bias_v* LR / BATCH_SIZE;
			d_bias_h_ave += d_bias_h* LR / BATCH_SIZE;


#ifdef DIAGNOSTIC

			//Compute Error
			NArray diff = reconstruction - visible;
			sqrdiff = Elewise::Mult(diff,diff);
			NArray sum0 = sqrdiff.Sum(0).Sum(0);
			mi.SetDevice(cpu);
			float error = sum0.Sum() / sqrdiff.Size().Prod();
			mse += error;
#endif
			if (i_batch % SYNCHRONIZATION_RATE == 0) {
				mi.WaitForAll();
			}

		}// === End batches for this epoch===
		mi.WaitForAll();
#ifdef DIAGNOSTIC
		mse = mse / n_batches;
		printf("MSE: %f\n", mse);

		//Look for update histogram problems
		mi.SetDevice(cpu);
		d_weights_ave /= n_batches;
		d_bias_v_ave /= n_batches;
		d_bias_h_ave /= n_batches;
		NArray weight_hist = d_weights_ave.Histogram(10);
		NArray bias_v_hist = d_bias_v_ave.Histogram(10);
		NArray bias_h_hist = d_bias_h_ave.Histogram(10);
		std::cout << "Weight Deltas:\n";
		weight_hist.ToStream(std::cout,ff);
		std::cout << "Visible Bias Deltas:\n";
		bias_v_hist.ToStream(std::cout,ff);
		std::cout << "Hidden Bias Deltas:\n";
		bias_h_hist.ToStream(std::cout,ff);

		//write an error side by side img
		if(has_gpu) {
			mi.SetDevice(gpu);
			NArray vis = Slice(visible,1,0,1);
			NArray rec = Slice(reconstruction,1,0,1);
			NArray sdif = Slice(sqrdiff,1,0,1);
			NArray error_side_by_side = Concat( {vis.Trans(),rec.Trans(),sdif.Trans()},0);
			printf("lkdsj\n");
			ofstream errof;
			errof.open(output_base +"error_img"+std::to_string(i_epoch), std::ifstream::out);
			error_side_by_side.ToStream(errof,ff);
			errof.close();
		}

		//write the hidden probabilities
		ofstream hof;
		hof.open(output_base +"_p_h_over_batch_e"+std::to_string(i_epoch), std::ifstream::out);
		Scale scale = hidden.Size();
		hof  << scale[0] << " " << scale[1] << " " << 1 << "\n";
		hidden.ToStream(hof, ff);
		hof.close();

		//write the current weights
		ofstream wof;
		wof.open(output_base +"_weights_e"+std::to_string(i_epoch), std::ifstream::out);
		scale = weights.Size();
		int x = (int)sqrt(scale[1]);
		int y = scale[1]/x;
		wof  << x << " " << y << " " << scale[0] << "\n";
		weights.Trans().ToStream(wof, ff);
		wof.close();
#endif
	}			//End epochs
	mi.PrintProfilerResults();
	ofstream of;
	of.open(output_base +"_weights_final", std::ifstream::out);
	Scale scale = weights.Size();
	int x = (int)sqrt(scale[1]);
	int y = scale[1]/x;
	of  << x << " " << y << " " << scale[0] << "\n";
	weights.ToStream(of, ff);
	of.close();

	return 0;

}

