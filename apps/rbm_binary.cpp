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


NArray propUp(NArray& visible, NArray& weights, NArray& bias) {
	NArray in_h = weights * visible + bias;
	return (1.0 / (1.0 + Elewise::Exp(-in_h))); // H x B
}

NArray propDown(NArray& hidden, NArray& weights, NArray& bias) {
	NArray in_v = weights.Trans() * hidden + bias;
	return (1.0 / (1.0 + Elewise::Exp(-in_v))); //V x B
}

NArray sample(NArray& ar) {
	NArray uniform_randoms = NArray::RandUniform(ar.Size(), 1.0);
	return ar < uniform_randoms; //H x B
}

NArray gibbsChain(NArray visible, NArray weights, NArray bias_v, NArray bias_h, int steps, bool sample_hiddens, bool sample_visibles) {

	NArray hidden, vis;
	vis = visible;
	for (int gibbs_step = 0; gibbs_step < steps; gibbs_step++) {

		//Propogate up to hiddens.  Sample visibles if specified
		if (sample_visibles) {
			vis = sample(vis);
			hidden = propUp(vis, weights, bias_h);
		} else {
			hidden = propUp(vis, weights, bias_h);
		}

		//Create a reconstruction. Sample Hiddens if specified.
		if (sample_hiddens) {
			hidden = sample(hidden);
			vis = propDown(hidden, weights, bias_v);
		} else {
			vis = propDown(hidden, weights, bias_v);
		}
	}
	if (sample_visibles) {
		vis = sample(vis);
	}
	return vis;

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
	float lr_s = params.sparsity_learning_rate();
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
	MnistData mdp(params.train_data_filename().c_str(), 0.9);
	DataProvider& dp = mdp;
	n_samples = dp.n_train_samples();
	sample_size = dp.sample_size();
	int n_batches = n_samples / batch_size;
	printf("\t%d samples of size %d\n", n_samples, sample_size);

	//Initialize arrays
	printf("Initialize data structures\n");
	NArray weights = NArray::Randn( { n_hidden, sample_size }, 0, .01);  //H x V



	//Get mean visible
	shared_ptr<float> train_set_raw = dp.next_batch(n_samples);
	NArray train_set = NArray::MakeNArray( { sample_size, n_samples }, train_set_raw); //V x B
	NArray bias_v = train_set.Sum(1)/n_samples;
	NArray bias_h;
	if (sparsity){
		bias_h = log(sparsity_target/(1-sparsity_target))*NArray::Ones( {n_hidden, 1 });
	}else{
		bias_h = NArray::Zeros( { n_hidden, 1 });
	}

	NArray d_weights = NArray::Zeros( { n_hidden, sample_size, });
	NArray d_bias_v = NArray::Zeros( { sample_size, 1 });
	NArray d_bias_h = NArray::Zeros( { n_hidden, 1 });

	NArray d_weights_ave = NArray::Zeros( { n_hidden, sample_size, });
	NArray d_bias_v_ave = NArray::Zeros( { sample_size, 1 });
	NArray d_bias_h_ave = NArray::Zeros( { n_hidden, 1 });
	NArray q_old = NArray::Zeros( { n_hidden, 1 });
	NArray zero_bias = NArray::Zeros( { n_hidden, 1 });
	NArray sqrdiff, p_visible, sampled_visible, p_hidden, sampled_hidden;
	NArray p_hidden_over_set = NArray::Zeros( {n_hidden,1});
	shared_ptr<float> eval_train_batch_raw = dp.next_batch(1000);
	NArray visible_t = NArray::MakeNArray( {sample_size, 1000}, eval_train_batch_raw);
	shared_ptr<float> eval_val_batch_raw = dp.next_val_batch(1000);
	NArray visible_val = NArray::MakeNArray( {sample_size, 1000}, eval_val_batch_raw);


	// ================ Begin training ================
	for (int i_epoch = 0; i_epoch < epochs; i_epoch++) {
		printf("Epoch %d\n", i_epoch);
		float mse = 0.0;
		p_hidden_over_set = NArray::Zeros( {n_hidden,1});
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
			shared_ptr<float> batch = dp.next_batch(batch_size);
			p_visible = NArray::MakeNArray( { sample_size, batch_size }, batch); //V x B

			//Initialize persistent chain if needed.
			if (persistent && !is_chain_init) {
				p_hidden = propUp(p_visible, weights, bias_h);
				sampled_hidden = sample(p_hidden);
				sampled_visible = propDown(sampled_hidden, weights, bias_v);
				is_chain_init = true;
			}

			//Apply momentum
			d_weights *= momentum;
			d_bias_v *= momentum;
			d_bias_h *= momentum;

			//Positive Phase
			p_hidden = propUp(p_visible, weights, bias_h); //H x B
			NArray d_weights_p = p_hidden * p_visible.Trans();
			NArray d_bias_h_p = p_hidden.Sum(1);
			NArray d_bias_v_p = p_visible.Sum(1);

			if(params.diag_hidden_activation_probability()){
				p_hidden_over_set += d_bias_h_p;
			}

			//Gather Sparsity statistics
			NArray d_weights_s, d_bias_h_s;
			if (sparsity) {
				mi.SetDevice(cpu);
				NArray q_current = d_bias_h_p / batch_size;  // H x 1
				NArray vis_ave = d_bias_v_p / batch_size; // V x 1
				NArray q_new = sparsity_decay * q_old + (1 - sparsity_decay) * q_current; //H x 1
				d_weights_s = (q_new - sparsity_target) * vis_ave.Trans(); // H x V
				d_bias_h_s = (q_new - sparsity_target);
				q_old = 1.0 * q_current;
				if (has_gpu) {
					mi.SetDevice(gpu);
				}
			}

			//perform Gibbs sampling.
			if (persistent) {
				sampled_visible = gibbsChain(sampled_visible, weights, bias_v, bias_h, gibbs_sampling_steps, sample_hiddens, sample_visibles);
				p_hidden = propUp(sampled_visible, weights, bias_h);

			} else {
				sampled_visible = gibbsChain(p_visible, weights, bias_v, bias_h, gibbs_sampling_steps, sample_hiddens, sample_visibles);
				p_hidden = propUp(sampled_visible, weights, bias_h);
			}

			//Negative Phase
			NArray d_weights_n = p_hidden * sampled_visible.Trans();
			//NArray d_bias_h_n = (propUp(sampled_visible, weights, zero_bias)).Sum(1);
			NArray d_bias_h_n = p_hidden.Sum(1);
			NArray d_bias_v_n = sampled_visible.Sum(1);

			//Update Weights
			d_weights += (d_weights_p - d_weights_n) * (lr / batch_size);
			d_bias_v += (d_bias_v_p - d_bias_v_n) * (lr / batch_size);
			d_bias_h += (d_bias_h_p - d_bias_h_n) * (lr / batch_size);
			weights += d_weights;
			bias_h += d_bias_h;
			bias_v += d_bias_v;
			if (sparsity) {
				weights -= d_weights_s*lr*lr_s;
				bias_h -= d_bias_h_s*lr*lr_s;
			}


			//Collect update statistics
			d_weights_ave += d_weights;
			d_bias_v_ave += d_bias_v;
			d_bias_h_ave += d_bias_h;
			if (params.diag_error()) {
				//Compute Error
				NArray diff = sampled_visible - p_visible;
				sqrdiff = Elewise::Mult(diff, diff);
				NArray sum0 = sqrdiff.Sum(0).Sum(0);
				mi.SetDevice(cpu);
				float error = sum0.Sum() / sqrdiff.Size().Prod();
				mse += error;
			}

			// Synchronize.
			if (i_batch % sync_period == 0) {
				mi.WaitForAll();
				//printf("bias_h (+,-,s): \n");
				//(d_bias_h_p* (lr / batch_size)).Histogram(5).ToStream(std::cout, ff);
				//(d_bias_h_n* (lr / batch_size)).Histogram(5).ToStream(std::cout, ff);
				//d_bias_h_s.Histogram(5).ToStream(std::cout, ff);
			}

		}  // End batches for this epoch
		mi.WaitForAll();

		//Diagnostics for this epoch
		if (params.diag_error()) {
			mse = mse / n_batches;
			printf("MSE: %f\n", mse);
		}
		if (params.diag_train_val_energy_diff()) {
			NArray hidden_t = propUp(visible_t, weights, bias_h);
			NArray hidden_val = propUp(visible_val, weights, bias_h);

			NArray E_t = -hidden_t.Trans() * bias_h - visible_t.Trans() * bias_v - Elewise::Mult((weights * visible_t), hidden_t).Sum(0).Trans();   // B X 1
			NArray E_val = -hidden_val.Trans() * bias_h - visible_val.Trans() * bias_v - Elewise::Mult((weights * visible_val), hidden_val).Sum(0).Trans(); // B X 1

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
			Scale scale = p_hidden.Size();
			Scale sout( { scale[0], 1, 1, 1 });
			writeNArray(p_hidden_over_set/n_samples, output_base + "_p_h_over_train_set_e" + std::to_string(i_epoch), sout);
		}

		if (params.diag_visible_recon_err()) {
			//write an error side by side img
			if (has_gpu) {
				mi.SetDevice(gpu);
				NArray vis = Slice(p_visible, 1, 0, 1);
				NArray rec = Slice(sampled_visible, 1, 0, 1);
				NArray sdif = Slice(sqrdiff, 1, 0, 1);
				NArray error_side_by_side = Concat( { vis.Trans(), rec.Trans(), sdif.Trans() }, 0);
				printf("lkdsj\n");
				ofstream errof;
				errof.open(output_base + "error_img" + std::to_string(i_epoch), std::ifstream::out);
				error_side_by_side.ToStream(errof, ff);
				errof.close();
			} else {
				Scale scale = p_visible.Size();
				int x = (int) sqrt(scale[0]);
				int y = scale[0] / x;
				Scale swrite( { x, y, 1, scale[1] });
				writeNArray(p_visible, output_base + "_vis_e" + std::to_string(i_epoch), swrite);
				writeNArray(sampled_visible, output_base + "_recon_e" + std::to_string(i_epoch), swrite);
				if (params.diag_error()) {
					writeNArray(sqrdiff, output_base + "_err_e" + std::to_string(i_epoch), swrite);
				}
			}
		}

		if (params.diag_epoch_weight_output()) {
			//write the current weights
			Scale wscale = weights.Size();
			int x = (int) sqrt(wscale[1]);
			int y = wscale[1] / x;
			Scale swrite( { x, y, 1, wscale[0] });
			writeNArray(weights.Trans(), output_base + "_weights_e" + std::to_string(i_epoch), swrite);
			Scale bvscale({x,y,1,1});
			writeNArray(bias_v, output_base + "_bias_v_e" + std::to_string(i_epoch), bvscale);
			Scale bhscale({1,1,1,n_hidden});
			writeNArray(bias_h, output_base + "_bias_h_e" + std::to_string(i_epoch), bhscale);
		}

	}   //End epochs
	mi.PrintProfilerResults();
	Scale scale = weights.Size();
	int x = (int) sqrt(scale[1]);
	int y = scale[1] / x;
	Scale swrite( { x, y, 1, scale[0] });
	writeNArray(weights.Trans(), output_base + "_weights_final", swrite);

	//Generate samples
	for (int i = 0; i < 5; i++) {
		sampled_visible = gibbsChain(sampled_visible, weights, bias_v, bias_h, 500, true, true);
		printf("Generated samples %d written.\n", i);
		Scale scale = sampled_visible.Size();
		int x = (int) sqrt(scale[0]);
		int y = scale[0] / x;
		Scale swrite( { x, y, 1, scale[1] });
		writeNArray(sample(sampled_visible), output_base + "_gen_" + std::to_string(i), swrite);
	}

	exit(0);

}

