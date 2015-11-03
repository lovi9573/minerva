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
#include <fcntl.h>
#include "mnist_raw.h"
#include "cifar_raw.h"
#include "rbmconfig.pb.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

using namespace minerva;

#define DIAGNOSTICS

//NArray must be such that a single sample is contiguous in memory.
void writeNArray(const NArray& array, std::string filename, Scale dims) {
	FileFormat ff;
	ff.binary = false;
	ofstream of;
	of.open(filename, std::ifstream::out);
	for (int d : dims) {
		of << d << " ";
	}
	of << "\n";
	array.ToStream(of, ff);
	of.close();
}

NArray propUp(NArray& visible, NArray& weights, NArray& bias){
	NArray in_h = weights * visible + bias;
	return ( 1.0 / (1.0 + Elewise::Exp(-in_h))); // H x B
}

NArray propDown(NArray& hidden, NArray& weights, NArray& bias){
	NArray in_v = weights.Trans() * hidden + bias;
	return ( 1.0 / (1.0 + Elewise::Exp(-in_v))); //V x B
}

NArray sample(NArray& ar){
	NArray uniform_randoms = NArray::RandUniform(ar.Size(), 1.0);
	return  ar > uniform_randoms; //H x B
}

int main(int argc, char** argv) {
	if (argc != 2) {
		printf("Use: rbm_binary  <path to config prototxt>\n");
		exit(0);
	}

	//Read in config and init vars
	rbm::RbmParameters params;
	int fin = open(argv[1], O_RDONLY);
	google::protobuf::io::FileInputStream param_fin(fin);
	google::protobuf::TextFormat::Parse(&param_fin, &params);
	FileFormat ff;
	ff.binary = false;
	int n_hidden = params.num_hidden();
	int epochs = params.epochs();
	int batch_size = params.batch_size();
	float momentum = params.momentum();
	float lr = params.learning_rate();
	int gibbs_sampling_steps = params.gibbs_sampling_steps();
	int sync_period = params.synchronization_period();
	std::string output_base = params.output_filename_base();
	bool persistent = params.persistent_gibbs_chain();
	bool sample_visibles = params.sample_visibles();
	bool sample_hiddens = params.sample_hiddens();
	bool sparsity = params.use_sparsity_target();
	float sparsity_target = params.sparsity_target();
	float sparsity_decay = params.sparsity_decay();
	bool is_chain_init = false;

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

	//Create training data provider
	printf("opening training data\n");
	int n_samples, sample_size;
	MnistData dp(params.train_data_filename().c_str(), 0.9);
	n_samples = dp.n_train_samples();
	sample_size = dp.SampleSize();
	int n_batches = n_samples / batch_size;
	printf("\t%d samples of size %d\n", n_samples, sample_size);

	//Initialize arrays
	printf("Initialize data structures\n");
	NArray weights = NArray::Randn( { n_hidden, sample_size }, 0, .01);  //H x V
	NArray bias_v = NArray::Zeros( { sample_size, 1 });
	NArray bias_h = NArray::Zeros( { n_hidden, 1 });

	NArray d_weights = NArray::Zeros( { n_hidden, sample_size, });
	NArray d_bias_v = NArray::Zeros( { sample_size, 1 });
	NArray d_bias_h = NArray::Zeros( { n_hidden, 1 });

	NArray d_weights_ave = NArray::Zeros( { n_hidden, sample_size, });
	NArray d_bias_v_ave = NArray::Zeros( { sample_size, 1 });
	NArray d_bias_h_ave = NArray::Zeros( { n_hidden, 1 });
	NArray sqrdiff, visible, reconstruction, hidden, sampled_hiddens, chain_visible;
	NArray q_old = NArray::Zeros( { n_hidden, 1 });

	// ================ Begin training ================
	for (int i_epoch = 0; i_epoch < epochs; i_epoch++) {
		printf("Epoch %d\n", i_epoch);
		float mse = 0.0;
		if (params.diag_error()) {
			d_weights_ave = NArray::Zeros( { n_hidden, sample_size, });
			d_bias_v_ave = NArray::Zeros( { sample_size, 1 });
			d_bias_h_ave = NArray::Zeros( { n_hidden, 1 });
		}
		//Begin batch
		for (int i_batch = 0; i_batch < n_batches; i_batch++) {
			if (has_gpu) {
				mi.SetDevice(gpu);
			}
			if (i_batch % sync_period == 0) {
				printf("\t Batch %d/%d\n", i_batch, n_batches);
			}

			NArray in_h, in_v;

			//Get minibatch
			shared_ptr<float> batch = dp.get_next_batch(batch_size);
			visible = NArray::MakeNArray( { sample_size, batch_size }, batch); //V x B

			//Initialize persistent chain if needed.
			if (persistent && !is_chain_init) {
				hidden = propUp(visible, weights, bias_h);
				sampled_hiddens = sample(hidden);
				chain_visible = propDown(sampled_hiddens,weights, bias_v);
				is_chain_init = true;
			}

			//Apply momentum
			d_weights *= momentum;
			d_bias_v *= momentum;
			d_bias_h *= momentum;

			//Positive Phase
			hidden = propUp(visible, weights, bias_h);
			NArray d_weights_p = hidden * visible.Trans();
			NArray d_bias_v_p = visible.Sum(1);
			NArray d_bias_h_p = hidden.Sum(1);

			//Gather Sparsity statistics
			NArray d_weights_s, d_bias_h_s;
			if (sparsity) {
				NArray q_current = hidden.Sum(1) / batch_size;  // H x 1
				NArray vis_ave = visible.Sum(1) / batch_size; // V x 1
				NArray q_new = (sparsity_decay * q_old + (1 - sparsity_decay) * q_current); //H x 1
				d_weights_s = (q_new - sparsity_target) * vis_ave.Trans(); // H x V
				d_bias_h_s = (q_new - sparsity_target);
				q_old = 1.0 * q_current;
			}

			//Setup for persistent Gibbs sampling.
			if (persistent) {
				hidden = propUp(chain_visible, weights, bias_h);
			}

			//Gibbs Sampling
			for (int gibbs_step = 0; gibbs_step < gibbs_sampling_steps; gibbs_step++) {

				//Create a reconstruction. Sample Hiddens if specified.
				if (sample_hiddens) {
					sampled_hiddens = sample(hidden);
					reconstruction = propDown(sampled_hiddens,weights,bias_v);
				} else {
					reconstruction = propDown(hidden,weights,bias_v);
				}


				//Propogate up to hiddens.  Sample visibles if specified
				if (sample_visibles) {
					NArray sampled_visibles = sample(reconstruction);
					hidden = propUp(sampled_visibles, weights, bias_h);
				} else {
					hidden = propUp(reconstruction, weights, bias_h);
				}
			}

			if (persistent) {
				chain_visible = 1.0 * reconstruction;
			}

			//Negative Phase
			NArray d_weights_n = hidden * reconstruction.Trans();
			NArray d_bias_v_n = reconstruction.Sum(1);
			NArray d_bias_h_n = hidden.Sum(1);

			//Update Weights
			d_weights += (d_weights_p - d_weights_n)* lr / batch_size;
			d_bias_v += (d_bias_v_p - d_bias_v_n)* lr / batch_size;
			d_bias_h += (d_bias_h_p - d_bias_h_n)* lr / batch_size;
			weights += d_weights ;
			bias_v += d_bias_v ;
			bias_h += d_bias_h ;
			if (sparsity) {
				weights -= d_weights_s;
				bias_h -= d_bias_h_s;
			}

			//Collect update statistics
			d_weights_ave += d_weights ;
			d_bias_v_ave += d_bias_v ;
			d_bias_h_ave += d_bias_h ;
			if (params.diag_error()) {
				//Compute Error
				NArray diff = reconstruction - visible;
				sqrdiff = Elewise::Mult(diff, diff);
				NArray sum0 = sqrdiff.Sum(0).Sum(0);
				mi.SetDevice(cpu);
				float error = sum0.Sum() / sqrdiff.Size().Prod();
				mse += error;
			}

			// Synchronize.
			if (i_batch % sync_period == 0) {
				mi.WaitForAll();
			}

		}  // End batches for this epoch
		mi.WaitForAll();

		//Diagnostics for this epoch
		if (params.diag_error()) {
			mse = mse / n_batches;
			printf("MSE: %f\n", mse);
		}
		if (params.diag_train_val_energy_diff()) {
			shared_ptr<float> batch_t = dp.get_next_batch(batch_size);
			NArray visible_t = NArray::MakeNArray( { sample_size, batch_size }, batch_t); //V x B
			shared_ptr<float> batch_val = dp.get_next_validation_batch(batch_size);
			NArray visible_val = NArray::MakeNArray( { sample_size, batch_size }, batch_val); //V x B

			NArray hidden_t = propUp(visible_t, weights, bias_h);
			NArray hidden_val = propUp(visible_val, weights, bias_h);

			NArray E_t = 	-hidden_t.Trans() * bias_h
						 	-visible_t.Trans() * bias_v
							-Elewise::Mult((weights * visible_t), hidden_t).Sum(0).Trans();   // B X 1
			NArray E_val = 	-hidden_val.Trans() * bias_h
							-visible_val.Trans() * bias_v
							-Elewise::Mult((weights * visible_val), hidden_val).Sum(0).Trans(); // B X 1

			mi.SetDevice(cpu);
			float E_diff = (E_t - E_val).Sum() / batch_size;
			printf("Train - Validation Energy difference: %f\n", E_diff);
		}

		if (params.diag_weight_update_hist()) {
			//Look for update histogram problems
			mi.SetDevice(cpu);
			d_weights_ave /= n_batches;
			d_bias_v_ave /= n_batches;
			d_bias_h_ave /= n_batches;
			NArray weight_hist = d_weights_ave.Histogram(10);
			NArray bias_v_hist = d_bias_v_ave.Histogram(10);
			NArray bias_h_hist = d_bias_h_ave.Histogram(10);
			std::cout << "Weight Deltas:\n";
			weight_hist.ToStream(std::cout, ff);
			std::cout << "Visible Bias Deltas:\n";
			bias_v_hist.ToStream(std::cout, ff);
			std::cout << "Hidden Bias Deltas:\n";
			bias_h_hist.ToStream(std::cout, ff);
		}

		if (params.diag_hidden_activation_probability()) {
			//write the hidden probabilities
			Scale scale = hidden.Size();
			Scale sout( { scale[0], scale[1], 1, 1 });
			writeNArray(hidden, output_base + "_p_h_over_batch_e" + std::to_string(i_epoch), sout);
		}

		if (params.diag_visible_recon_err()) {
			//write an error side by side img
			if (has_gpu) {
				mi.SetDevice(gpu);
				NArray vis = Slice(visible, 1, 0, 1);
				NArray rec = Slice(reconstruction, 1, 0, 1);
				NArray sdif = Slice(sqrdiff, 1, 0, 1);
				NArray error_side_by_side = Concat( { vis.Trans(), rec.Trans(), sdif.Trans() }, 0);
				printf("lkdsj\n");
				ofstream errof;
				errof.open(output_base + "error_img" + std::to_string(i_epoch), std::ifstream::out);
				error_side_by_side.ToStream(errof, ff);
				errof.close();
			}
		}

		if (params.diag_epoch_weight_output()) {
			//write the current weights
			Scale wscale = weights.Size();
			int x = (int) sqrt(wscale[1]);
			int y = wscale[1] / x;
			Scale swrite( { x, y, 1, wscale[0] });
			writeNArray(weights.Trans(), output_base + "_weights_e" + std::to_string(i_epoch), swrite);
		}

	}   //End epochs
	mi.PrintProfilerResults();
	Scale scale = weights.Size();
	int x = (int) sqrt(scale[1]);
	int y = scale[1] / x;
	Scale swrite( { x, y, 1, scale[0] });
	writeNArray(weights.Trans(), output_base + "_weights_final", swrite);
	return 0;

}

