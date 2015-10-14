/*
 * mnist_raw.h
 *
 *  Created on: Oct 14, 2015
 *      Author: jlovitt
 */

#ifndef APPS_MNIST_RAW_H_
#define APPS_MNIST_RAW_H_

#include "mnist_raw.h"
#include <cstdio>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <memory>

using namespace std;

class MnistData{
public:
	MnistData(char* data_filename);
	int BigtoLittle(int val);
	shared_ptr<float> GetNextBatch(int batchsize);
	int nSamples();
	int SampleSize();
private:
	int n_samples;
	int n_rows;
	int n_columns;
	ifstream datastream;
};


MnistData::MnistData(char* data_filename) {
	printf("opening \"%s\"\n",data_filename);
	datastream.open(data_filename, std::ifstream::binary);
	datastream.clear();
	datastream.seekg(0,std::ifstream::beg);
	char header[16];
	datastream.read(header, 16 * sizeof(char));
	int* iheader = reinterpret_cast<int*>(header);
	n_samples = BigtoLittle(iheader[1]);
	n_rows = BigtoLittle(iheader[2]);
	n_columns = BigtoLittle(iheader[3]);
	printf("file contains data of size %dx%dx%d\n",n_samples,n_rows,n_columns);
}

int MnistData::BigtoLittle(int val) {
	char out[4];
	char* valbytes = reinterpret_cast<char*>(&val);
	out[0] = valbytes[3];
	out[1] = valbytes[2];
	out[2] = valbytes[1];
	out[3] = valbytes[0];
	return *(reinterpret_cast<int*>(out));
}

shared_ptr<float> MnistData::GetNextBatch(int batchsize) {
	char buf[128];
	shared_ptr<float> data(new float[batchsize * n_rows * n_columns],
			[](float* ptr) {
				delete[] ptr;
			});
	int rd = 0;
	int idata = 0;
	while (batchsize > 0) {
		rd = max(128, batchsize);
		datastream.read(buf, rd * sizeof(char));
		for (int i = 0; i < rd; i++) {
			data.get()[idata++] = ((float) buf[i]) / 255.0f;
		}
		batchsize -= rd;
	}
	return data;
}

int MnistData::nSamples(){
	return n_samples;
}


int MnistData::SampleSize(){
	return n_rows*n_columns;
}


#endif /* APPS_MNIST_RAW_H_ */
