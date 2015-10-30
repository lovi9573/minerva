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

int main(int argc, char** argv) {
	if (argc != 3) {
		printf(
				"Use: rbm_binary <path to input data> <path to config prototxt>\n");
		exit(0);
	}

	//Read in config and init vars
	rbm::RbmParameters params;
	int fin = open(argv[2], O_RDONLY);
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
	bool is_chain_init = false;
	int convolution_stride = 3;
	int convolution_patch_dim = 5;
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
	int n_samples, n_channels, dim_y,dim_x;
	CifarData dp(argv[1],0.9);
	n_samples = dp.nTrainSamples();
	n_channels = dp.nTrainChannels();
	dim_y = dp.dim_y();
	dim_x = dp.dim_x();
	int n_batches = n_samples / batch_size;
	printf("\t%d samples of size %d\n", n_samples, sample_size);

	//Initialize data structures
	printf("Initialize data structures\n");
	int hidden_dim_x = (dim_x+2*convolution_padding-convolution_patch_dim)/convolution_stride;
	int hidden_dim_y = (dim_y+2*convolution_padding-convolution_patch_dim)/convolution_stride;
	Scale weight_scale( { convolution_patch_dim, convolution_patch_dim, n_channels,_n_hidden });
	Scale sample_scale({dim_x, dim_y, n_channels, batch_size});
	Scale visible_bias_scale({dim_x, dim_y, n_channels});
	Scale hidden_bias_scale({hidden_dim_x, hidden_dim_y, n_hidden});
	Scale hidden_scale({ hidden_dim_x, hidden_dim_y, n_hidden, batch_size});
	ConvInfo fwd_conv_info(convolution_padding,convolution_padding,convolution_stride,convolution_stride);

	//Initialize arrays
	NArray weights = NArray::Randn(weight_scale, 0, .01);  //H x V
	NArray bias_v = NArray::Zeros( visible_bias_scale);
	NArray bias_h = NArray::Zeros( hidden_bias_scale);

	NArray d_weights = NArray::Zeros( weight_scale);
	NArray d_bias_v = NArray::Zeros( { sample_size, 1 });
	NArray d_bias_h = NArray::Zeros( { n_hidden, 1 });

	NArray d_weights_ave = NArray::Zeros( weight_scale);
	NArray d_bias_v_ave = NArray::Zeros( { sample_size, 1 });
	NArray d_bias_h_ave = NArray::Zeros( { n_hidden, 1 });
	NArray sqrdiff, visible, reconstruction, hidden, sampled_hiddens,
			chain_visible;

	//Begin training
	for (int i_epoch = 0; i_epoch < epochs; i_epoch++) {
		printf("Epoch %d\n", i_epoch);
		float mse = 0.0;
		if (params.diag_error()) {
			d_weights_ave = NArray::Zeros(weight_scale);
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
			shared_ptr<float> batch = dp.GetNextBatch(batch_size);
			visible = NArray::MakeNArray( sample_scale, batch); //V x B

			//Initialize persistent chain if needed.
			if (persistent && !is_chain_init) {
				in_h = weights * visible + bias_h;
				hidden = 1.0 / (1.0 + Elewise::Exp(-in_h)); // H x B
				NArray uniform_randoms = NArray::RandUniform(hidden.Size(),
						1.0);
				sampled_hiddens = hidden > uniform_randoms; //H x B
				in_v = weights.Trans() * sampled_hiddens + bias_v;
				chain_visible = 1.0 / (1.0 + Elewise::Exp(-in_v)); //V x B
				is_chain_init = true;
			}

			//Apply momentum
			d_weights *= momentum;
			d_bias_v *= momentum;
			d_bias_h *= momentum;

			//Positive Phase
			in_h = Convolution::ConvForward(visible, weights, bias_h, fwd_conv_info);
			//in_h = weights * visible + bias_h;
			hidden = 1.0 / (1.0 + Elewise::Exp(-in_h)); // H x B

			NArray d_weights_p = NArray.Zeros(weight_scale);
			d_weights_p = Convolution::ConvBackwardFilter(hidden,visible,weights,fwd_conv_info);
			//NArray d_weights_p = hidden * visible.Trans();
			NArray d_bias_v_p = visible.Sum(visible.Size().NumDims()-1);
			NArray d_bias_h_p = hidden.Sum(hidden.Size().NumDims()-1);

			//Setup for Gibbs sampling.
			if (persistent) {
				in_h = Convolution::ConvForward(chain_visible, weights, bias_h, fwd_conv_info);
				//in_h = weights * chain_visible + bias_h;
				hidden = 1.0 / (1.0 + Elewise::Exp(-in_h)); // H x B
			} else {
				NArray uniform_randoms = NArray::RandUniform(hidden.Size(),
						1.0);
				sampled_hiddens = hidden > uniform_randoms; //H x B
			}

			//Gibbs Sampling
			for (int gibbs_step = 0; gibbs_step < gibbs_sampling_steps;
					gibbs_step++) {
				//Create a reconstruction. Sample Hiddens if specified.
				if (sample_hiddens) {
					NArray uniform_randoms = NArray::RandUniform(hidden.Size(),
							1.0);
					sampled_hiddens = hidden > uniform_randoms; //H x B
					in_v = Convolution::ConvForward(sampled_hiddens,weights, bias_v, fwd_conv_info);
					//in_v = weights.Trans() * sampled_hiddens + bias_v;
				} else {
					in_v = Convolution::ConvForward(hidden,weights, bias_v, fwd_conv_info);
					//in_v = weights.Trans() * hidden + bias_v;
				}
				reconstruction = 1.0 / (1.0 + Elewise::Exp(-in_v)); //V x B

				//Propogate up to hiddens.  Sample visibles if specified
				if (sample_visibles) {
					NArray uniform_randoms = NArray::RandUniform(
							reconstruction.Size(), 1.0);
					NArray sampled_visibles = reconstruction > uniform_randoms;
					in_h = Convolution::ConvForward(sampled_visibles,weights, bias_h, fwd_conv_info);
					//in_h = weights * sampled_visibles + bias_h;
				} else {
					in_h = Convolution::ConvForward(reconstruction,weights, bias_h, fwd_conv_info);
					//in_h = weights * reconstruction + bias_h;
				}
				hidden = 1.0 / (1.0 + Elewise::Exp(-in_h));  //H x B
			}
			if (persistent) {
				chain_visible = 1.0 * reconstruction;
			}

			//Negative Phase
			NArray d_weights_n = NArray.Zeros(weight_scale);
			d_weights_p = Convolution::ConvBackwardFilter(hidden,reconstruction,weights,fwd_conv_info);
			//NArray d_weights_p = hidden * visible.Trans();
			NArray d_bias_v_n = reconstruction.Sum(visible.Size().NumDims()-1);
			NArray d_baias_h_n = hidden.Sum(hidden.Size().NumDims()-1);
			/*NArray d_weights_n = hidden * reconstruction.Trans();
			NArray d_bias_v_n = reconstruction.Sum(1);
			NArray d_baias_h_n = hidden.Sum(1);
*/
			//Update Weights
			d_weights += (d_weights_p - d_weights_n);
			d_bias_v += (d_bias_v_p - d_bias_v_n);
			d_bias_h += (d_bias_h_p - d_bias_h_n);

			weights += d_weights * lr / batch_size;
			bias_v += d_bias_v * lr / batch_size;
			bias_h += d_bias_h * lr / batch_size;

			d_weights_ave += d_weights * lr / batch_size;
			d_bias_v_ave += d_bias_v * lr / batch_size;
			d_bias_h_ave += d_bias_h * lr / batch_size;

			if (params.diag_error()) {

				//Compute Error
				NArray diff = reconstruction - visible;
				sqrdiff = Elewise::Mult(diff, diff);
				NArray sum0 = sqrdiff.Sum(0);
				while(sum0.Size().NumDims() > 1){
					sum0 = sum0.Sum(0);
				}
				mi.SetDevice(cpu);
				float error = sum0.Sum() / sqrdiff.Size().Prod();
				mse += error;
			}
			if (i_batch % sync_period == 0) {
				mi.WaitForAll();
			}

		}// End batches for this epoch
		mi.WaitForAll();

		//Diagnostics for this epoch
		if (params.diag_error()) {
			mse = mse / n_batches;
			printf("MSE: %f\n", mse);
		}
		if (params.diag_train_val_energy_diff()){
			shared_ptr<float> batch_t = dp.GetNextBatch(batch_size);
			NArray visible_t = NArray::MakeNArray( { sample_size, batch_size }, batch_t); //V x B
			shared_ptr<float> batch_val = dp.GetNextBatch(batch_size);
			NArray visible_val = NArray::MakeNArray( { sample_size, batch_size }, batch_val); //V x B

			NArray in_h = Convolution::ConvForward(visible_t,weights, bias_h, fwd_conv_info);
			//NArray in_h = weights * visible_t + bias_h;
			NArray hidden_t = 1.0 / (1.0 + Elewise::Exp(-in_h)); // H x B
			in_h = Convolution::ConvForward(visible_val,weights, bias_h, fwd_conv_info);
			//in_h = weights * visible_val + bias_h;
			NArray hidden_val = 1.0 / (1.0 + Elewise::Exp(-in_h)); // H x B

			//TODO(jesselovitt): Compute Energy of both == Convert to convolution from here down====
			NArray E_t = hidden_t.Trans()*bias_h + visible_t.Trans()*bias_v + Elewise::Mult((weights * visible_t), hidden_t).Sum(0).Trans();   // B X 1
			NArray E_val = hidden_val.Trans()*bias_h + visible_val.Trans()*bias_v + Elewise::Mult((weights * visible_val), hidden_val).Sum(0).Trans();   // B X 1

			mi.SetDevice(cpu);
			float E_diff = (E_t - E_val).Sum()/batch_size;
			printf("Validation - train Energy difference: %f\n",E_diff);
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
			Scale sout( { scale[0], scale[1], 1 });
			writeNArray(hidden,
					output_base + "_p_h_over_batch_e" + std::to_string(i_epoch),
					sout);
		}

		if (params.diag_visible_recon_err()) {
			//write an error side by side img
			if (has_gpu) {
				mi.SetDevice(gpu);
				NArray vis = Slice(visible, 1, 0, 1);
				NArray rec = Slice(reconstruction, 1, 0, 1);
				NArray sdif = Slice(sqrdiff, 1, 0, 1);
				NArray error_side_by_side = Concat( { vis.Trans(), rec.Trans(),
						sdif.Trans() }, 0);
				printf("lkdsj\n");
				ofstream errof;
				errof.open(output_base + "error_img" + std::to_string(i_epoch),
						std::ifstream::out);
				error_side_by_side.ToStream(errof, ff);
				errof.close();
			}
		}

		if (params.diag_epoch_weight_output()) {
			//write the current weights
			Scale wscale = weights.Size();
			int x = (int) sqrt(wscale[1]);
			int y = wscale[1] / x;
			Scale swrite( { x, y, wscale[0] });
			writeNArray(weights.Trans(),
					output_base + "_weights_e" + std::to_string(i_epoch),
					swrite);
		}

	}//End epochs
	mi.PrintProfilerResults();
	Scale scale = weights.Size();
	int x = (int) sqrt(scale[1]);
	int y = scale[1] / x;
	Scale swrite( { x, y, scale[0] });
	writeNArray(weights.Trans(), output_base + "_weights_final", swrite);
	return 0;

}

