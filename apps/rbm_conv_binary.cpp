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


NArray propUp(NArray& visible, NArray& weights, NArray& bias, ConvInfo conv_info) {
	NArray in_h =Convolution::ConvForward(visible, weights, bias, conv_info);
	return (1.0 / (1.0 + Elewise::Exp(-in_h))); // H x B
}

NArray propDown(NArray& hidden, NArray& weights, NArray& visible, NArray& bias, ConvInfo conv_info) {
	NArray in_v = Convolution::ConvBackwardData(hidden, visible, weights, conv_info);
	in_v = in_v + bias;
	return (1.0 / (1.0 + Elewise::Exp(-in_v))); //V x B
}

NArray sample(NArray& ar) {
	NArray uniform_randoms = NArray::RandUniform(ar.Size(), 1.0);
	return ar < uniform_randoms; //H x B
}

NArray gibbsChain(NArray visible, NArray weights, NArray bias_v, NArray bias_h, ConvInfo conv_info, int steps, bool sample_hiddens, bool sample_visibles) {

	NArray hidden, vis;
	vis = visible;
	for (int gibbs_step = 0; gibbs_step < steps; gibbs_step++) {

		//Propogate up to hiddens.  Sample visibles if specified
		if (sample_visibles) {
			vis = sample(vis);
			hidden = propUp(vis, weights, bias_h, conv_info);
		} else {
			hidden = propUp(vis, weights, bias_h, conv_info);
		}

		//Create a reconstruction. Sample Hiddens if specified.
		if (sample_hiddens) {
			hidden = sample(hidden);
			vis = propDown(hidden, weights,vis, bias_v, conv_info);
		} else {
			vis = propDown(hidden, weights, vis, bias_v, conv_info);
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
	int convolution_stride = 1;
	int convolution_patch_dim = 28;
	int convolution_padding = 0;

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
	int n_samples, n_channels,dim_y,dim_x;
	MnistData dp(params.train_data_filename().c_str(), 0.9);
	n_samples = dp.n_train_samples();
	//sample_size = dp.SampleSize();
	n_channels = dp.n_channels();
	dim_y = dp.dim_y();
	dim_x = dp.dim_x();
	int n_batches = n_samples / batch_size;
	printf("\t%d samples of size %dx%dx%d\n", n_samples, dim_x,dim_y,n_channels);

	//Initialize arrays
	printf("Initialize data structures\n");
	int hidden_dim_x = (dim_x+2*convolution_padding-convolution_patch_dim)/convolution_stride + 1;
	int hidden_dim_y =  (dim_y+2*convolution_padding-convolution_patch_dim)/convolution_stride + 1;
	Scale weight_scale( { convolution_patch_dim, convolution_patch_dim, n_channels, n_hidden });
	Scale sample_scale({dim_x, dim_y, n_channels, batch_size});
	Scale visible_bias_scale({dim_x, dim_y,n_channels,1});
	Scale hidden_bias_scale({ n_hidden});
	Scale hidden_bias_2_hidden({ 1,1,n_hidden,1});
	Scale hidden_scale({ hidden_dim_x, hidden_dim_y, n_hidden, batch_size});
	Scale hidden_prob_scale({ hidden_dim_x, hidden_dim_y, n_hidden, 1});
	ConvInfo fwd_conv_info(convolution_padding,convolution_padding,convolution_stride,convolution_stride);

	//Initialize arrays
	NArray weights = NArray::Randn(weight_scale, 0, .01);  //H x V

	//Get mean visible
	shared_ptr<float> train_set_raw = dp.get_next_batch(n_samples);
	NArray train_set = NArray::MakeNArray( sample_scale, train_set_raw); //V x B
	NArray bias_v = train_set.Sum(3)/n_samples;
	NArray bias_h = NArray::Zeros( hidden_bias_scale);

	NArray d_weights = NArray::Zeros( weight_scale);
	NArray d_bias_v = NArray::Zeros( visible_bias_scale);
	NArray d_bias_h = NArray::Zeros( hidden_bias_scale);

	NArray d_weights_ave = NArray::Zeros( weight_scale);
	NArray d_bias_v_ave = NArray::Zeros( visible_bias_scale);
	NArray d_bias_h_ave = NArray::Zeros( hidden_bias_scale);

	NArray q_old = NArray::Zeros( hidden_bias_2_hidden);
	NArray zero_bias = NArray::Zeros( hidden_bias_scale);
	NArray sqrdiff, p_visible, sampled_visible, p_hidden, sampled_hidden;
	NArray p_hidden_over_set = NArray::Zeros( hidden_prob_scale);
	shared_ptr<float> eval_train_batch_raw = dp.get_next_batch(1000);
	NArray visible_t = NArray::MakeNArray( {dim_x, dim_y, n_channels, 1000}, eval_train_batch_raw);
	shared_ptr<float> eval_val_batch_raw = dp.get_next_validation_batch(1000);
	NArray visible_val = NArray::MakeNArray( {dim_x, dim_y, n_channels, 1000}, eval_val_batch_raw);


	// ================ Begin training ================
	for (int i_epoch = 0; i_epoch < epochs; i_epoch++) {
		printf("Epoch %d\n", i_epoch);
		float mse = 0.0;
		p_hidden_over_set = NArray::Zeros( hidden_prob_scale);
		if (params.diag_error()) {
			d_weights_ave = NArray::Zeros(weight_scale);
			d_bias_v_ave = NArray::Zeros( visible_bias_scale);
			d_bias_h_ave = NArray::Zeros( hidden_bias_scale);
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
			p_visible = NArray::MakeNArray( sample_scale, batch); //V x B

			//Initialize persistent chain if needed.
			if (persistent && !is_chain_init) {
				p_hidden = propUp(p_visible, weights, bias_h,fwd_conv_info);
				sampled_hidden = sample(p_hidden);
				sampled_visible = propDown(sampled_hidden, weights,p_visible, bias_v,fwd_conv_info);
				is_chain_init = true;
			}

			//Apply momentum
			d_weights *= momentum;
			d_bias_v *= momentum;
			d_bias_h *= momentum;
			//Positive Phase
			p_hidden = propUp(p_visible, weights, bias_h,fwd_conv_info); //H x B
			NArray d_weights_p = Convolution::ConvBackwardFilter(p_hidden,p_visible,weights,fwd_conv_info);
			NArray d_bias_h_p = p_hidden.Sum(p_hidden.Size().NumDims()-1).Sum(0).Sum(1);
			NArray d_bias_v_p = p_visible.Sum(p_visible.Size().NumDims()-1).Sum(0).Sum(1);
//std::cout << p_hidden_over_set.Size() << "\n" << p_hidden.Size();
			if(params.diag_hidden_activation_probability()){
				p_hidden_over_set += p_hidden.Sum(3);
			}

			//Gather Sparsity statistics
			NArray d_weights_s, d_bias_h_s;
			if (sparsity) {
				mi.SetDevice(cpu);
				NArray q_current = d_bias_h_p / batch_size;  // H x 1
				NArray vis_ave = d_bias_v_p / batch_size; // V x 1
				NArray q_new = sparsity_decay * q_old + (1 - sparsity_decay) * q_current; //H x 1
				//d_weights_s = (q_new - sparsity_target) * vis_ave.Trans(); // H x V
				//d_bias_h_s = (q_new - sparsity_target);
				q_old = 1.0 * q_current;
				if (has_gpu) {
					mi.SetDevice(gpu);
				}
			}

			//perform Gibbs sampling.
			if (persistent) {
				sampled_visible = gibbsChain(sampled_visible, weights, bias_v, bias_h,fwd_conv_info, gibbs_sampling_steps, sample_hiddens, sample_visibles);
				p_hidden = propUp(sampled_visible, weights, bias_h,fwd_conv_info);

			} else {
				sampled_visible = gibbsChain(p_visible, weights, bias_v, bias_h,fwd_conv_info, gibbs_sampling_steps, sample_hiddens, sample_visibles);
				p_hidden = propUp(sampled_visible, weights, bias_h,fwd_conv_info);
			}
			//Negative Phase
			NArray d_weights_n =  Convolution::ConvBackwardFilter(p_hidden,sampled_visible,weights,fwd_conv_info);
			//NArray d_bias_h_n = (propUp(sampled_visible, weights, zero_bias,fwd_conv_info)).Sum(1);
			NArray d_bias_h_n = p_hidden.Sum(p_hidden.Size().NumDims()-1).Sum(0).Sum(1);
			NArray d_bias_v_n = sampled_visible.Sum(sampled_visible.Size().NumDims()-1).Sum(0).Sum(1);
			//Update Weights
			d_weights += (d_weights_p - d_weights_n) * (lr / batch_size);
			d_bias_v += (d_bias_v_p - d_bias_v_n) * (lr / batch_size);
			d_bias_h += ((d_bias_h_p - d_bias_h_n) * (lr / batch_size)).Reshape(hidden_bias_scale);
			weights += d_weights;
			//bias_h += d_bias_h;
			//bias_v += d_bias_v;
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
			NArray hidden_t = propUp(visible_t, weights, bias_h,fwd_conv_info);
			NArray hidden_val = propUp(visible_val, weights, bias_h,fwd_conv_info);

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
			writeNArray(p_hidden_over_set/n_samples, output_base + "_p_h_over_train_set_e" + std::to_string(i_epoch), p_hidden_over_set.Size());
		}

		if (params.diag_visible_recon_err()) {
			//write an error side by side img
			writeNArray(p_visible, output_base + "_vis_e" + std::to_string(i_epoch), p_visible.Size());
			writeNArray(sampled_visible, output_base + "_recon_e" + std::to_string(i_epoch), sampled_visible.Size());
			if (params.diag_error()) {
				writeNArray(sqrdiff, output_base + "_err_e" + std::to_string(i_epoch), sqrdiff.Size());
			}
		}

		if (params.diag_epoch_weight_output()) {
			//write the current weights
			writeNArray(weights, output_base + "_weights_e" + std::to_string(i_epoch), weights.Size());
			writeNArray(bias_v, output_base + "_bias_v_e" + std::to_string(i_epoch), bias_v.Size());
			writeNArray(bias_h, output_base + "_bias_h_e" + std::to_string(i_epoch), hidden_bias_2_hidden);
		}

	}   //End epochs
	mi.PrintProfilerResults();
	writeNArray(weights, output_base + "_weights_final", weights.Size());

	//Generate samples
	for (int i = 0; i < 5; i++) {
		sampled_visible = gibbsChain(sampled_visible, weights, bias_v, bias_h,fwd_conv_info, 500, true, true);
		printf("Generated samples %d written.\n", i);
		writeNArray(sample(sampled_visible), output_base + "_gen_" + std::to_string(i), sampled_visible.Size());
	}

	exit(0);

}

