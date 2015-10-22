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
#include "rbmconfig.pb.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

using namespace minerva;





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
				"Use: rbm_binary <path to input data> <output filename base>\n");
		exit(0);
	}

	//Read in config and init vars
	rbm::RbmParameters params;
	int fin = open(argv[2],O_RDONLY);
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
	bool binary_visibles = params.binary_visibles();
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
	MnistData dp(argv[1]);
	n_samples = dp.nSamples();
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
	NArray sqrdiff, visible, reconstruction, hidden, chain_visible;

	//Begin training
	for (int i_epoch = 0; i_epoch < epochs; i_epoch++) {
		printf("Epoch %d\n", i_epoch);
#ifdef DIAGNOSTIC
		float mse = 0.0;
		d_weights_ave = NArray::Zeros( { n_hidden, sample_size, });
		d_bias_v_ave = NArray::Zeros( { sample_size, 1 });
		d_bias_h_ave = NArray::Zeros( { n_hidden, 1 });
#endif
		for (int i_batch = 0; i_batch < n_batches; i_batch++) {
			if (has_gpu) {
				mi.SetDevice(gpu);
			}
			if (i_batch % sync_period == 0) {
				printf("\t Batch %d/%d\n", i_batch, n_batches);
			}
			//Get minibatch
			shared_ptr<float> batch = dp.GetNextBatch(batch_size);
			visible = NArray::MakeNArray( { sample_size, batch_size }, batch); //V x B
			if(persistent && !is_chain_init){
				chain_visible = 1.0*visible;
				is_chain_init = true;
			}

			//Apply momentum
			d_weights *= momentum;
			d_bias_v *= momentum;
			d_bias_h *= momentum;

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

			for(int gibbs_step = 0; gibbs_step < gibbs_sampling_steps; gibbs_step++){
				//Sample Hiddens
				mi.SetDevice(cpu);
				NArray uniform_randoms = NArray::RandUniform(hidden.Size(), 1.0);
				if (has_gpu) {
					mi.SetDevice(gpu);
				}
				NArray sampled_hiddens = hidden > uniform_randoms; //H x B

				NArray in_v = weights.Trans() * sampled_hiddens + bias_v;
				reconstruction = 1.0 / (1.0 + Elewise::Exp(-in_v)); //V x B
				if(binary_visibles){
					mi.SetDevice(cpu);
					uniform_randoms = NArray::RandUniform(reconstruction.Size(),1.0);
					if (has_gpu) {
						mi.SetDevice(gpu);
					}
					NArray sampled_visibles = reconstruction > uniform_randoms;
					in_h = weights * sampled_visibles + bias_h;
				}else{
					in_h = weights * reconstruction + bias_h;
				}
				hidden = 1.0 / (1.0 + Elewise::Exp(-in_h));  //H x B
			}
			if(persistent){
				chain_visible = 1.0*reconstruction;
			}

			//Negative Phase
			NArray d_weights_n = hidden * reconstruction.Trans();
			NArray d_bias_v_n = reconstruction.Sum(1);
			NArray d_bias_h_n = hidden.Sum(1);

			//Update Weights
			d_weights += (d_weights_p - d_weights_n);
			d_bias_v += (d_bias_v_p - d_bias_v_n);
			d_bias_h += (d_bias_h_p - d_bias_h_n);

			weights += d_weights* lr / batch_size;
			bias_v += d_bias_v* lr / batch_size;
			bias_h += d_bias_h* lr / batch_size;

			d_weights_ave += d_weights* lr / batch_size;
			d_bias_v_ave += d_bias_v* lr / batch_size;
			d_bias_h_ave += d_bias_h* lr / batch_size;


#ifdef DIAGNOSTIC

			//Compute Error
			NArray diff = reconstruction - visible;
			sqrdiff = Elewise::Mult(diff,diff);
			NArray sum0 = sqrdiff.Sum(0).Sum(0);
			mi.SetDevice(cpu);
			float error = sum0.Sum() / sqrdiff.Size().Prod();
			mse += error;
#endif
			if (i_batch % sync_period == 0) {
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

#endif
		//write the current weights
		ofstream wof;
		wof.open(output_base +"_weights_e"+std::to_string(i_epoch), std::ifstream::out);
		Scale scale = weights.Size();
		int x = (int)sqrt(scale[1]);
		int y = scale[1]/x;
		wof  << x << " " << y << " " << scale[0] << "\n";
		weights.Trans().ToStream(wof, ff);
		wof.close();
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

