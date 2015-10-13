/*
 * rbm_binary.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: jlovitt
 */

#include <cstdio>
#include <minerva.h>

using namespace minerva;


#define DATA_MIN 0
#define DATA_MAX 255

#define N_HIDDEN 128
#define N_EPOCHS 10
#define BATCH_SIZE 64

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
	FILE* fdata = fopen(argv[0],"r");
	fscanf(fdata,"%i", &n_samples);
	fscanf(fdata,"%i", &sample_size);
	NArray training_dat = NArray::Zeros({n_samples,sample_size});
	float* training_dat_raw = training_dat.Get().get();
	for(int i = 0; i < n_samples*sample_size; i++){
		fscanf(fdata,"%f", &training_dat_raw[i]);
	}

	//Scale the data to  (0,1)
	training_dat = (training_dat - DATA_MIN) / (DATA_MAX - DATA_MIN);

	//Initialize arrays
	printf("Initialize data structures\n");
	NArray weights = NArray::Randn({sample_size, N_HIDDEN},0,1);
	NArray bias_v = NArray::Zeros({1,sample_size});
	NArray bias_h = NArray::Zeros({1,sample_size});

	NArray d_weights = NArray::Zeros({sample_size, N_HIDDEN});
	NArray d_bias_v = NArray::Zeros({1,sample_size});
	NArray d_bias_h = NArray::Zeros({1,sample_size});

	int n_batches = n_samples/BATCH_SIZE;

	//Begin training
	for(int i_epoch = 0; i_epoch< N_EPOCHS; i_epoch++){
		printf("Epoch %d\n",i_epoch);
		float mse = 0.0;
		for(int i_batch = 0; i_batch < n_batches; i_batch++){
			//NArray training_batch = training_dat.
		}//End batches

	}//End epochs

	return 0;

}

