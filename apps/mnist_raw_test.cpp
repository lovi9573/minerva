/*
 * mnist_raw_test.cpp
 *
 *  Created on: Oct 17, 2015
 *      Author: user
 */


#include <cstdio>
#include <minerva.h>
#include <iomanip>
#include <fstream>
#include <string>
#include "mnist_raw.h"

using namespace minerva;

#define BATCH_SIZE 20

int main(int argc, char**argv){

	FileFormat ff;
	ff.binary = false;

	//Load the training data
	printf("load data\n");
	int n_samples,sample_size;
	MnistData dp(argv[1]);
	n_samples = dp.nSamples();
	sample_size = dp.SampleSize();
	printf("%d samples of size %d\n",n_samples,sample_size);

	//Setup Minerva
	IMinervaSystem::Init(&argc, &argv);
	auto&& mi = IMinervaSystem::Interface();
	uint64_t cpu = mi.CreateCpuDevice();
	mi.SetDevice(cpu);

	//Get minibatch
	shared_ptr<float> batch = dp.GetNextBatch(BATCH_SIZE);
	n_samples = dp.nSamples();
	sample_size = dp.SampleSize();
	NArray nbatch = NArray::MakeNArray({sample_size,BATCH_SIZE}, batch);
	ofstream of;
	of.open(argv[2], std::ifstream::out);
	nbatch.Trans().ToStream(of, ff);
	of.close();
}
