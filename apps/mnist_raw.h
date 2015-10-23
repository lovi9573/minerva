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
	MnistData(char* data_filename,float);
	int BigtoLittle(int val);
	shared_ptr<float> GetNextBatch(int batchsize);
	shared_ptr<float> GetNextValidationBatch(int batchsize);
	void setSplit(float);
	int nTrainSamples();
	int nValSamples();
	int SampleSize();
private:
	int n_samples;
	int n_rows;
	int n_columns;
	float split_;
	int trainpos_;
	ifstream datastream;
	ifstream valstream;
};


MnistData::MnistData(char* data_filename, float split) {
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
	split_ = split;
	valstream.open(data_filename, std::ifstream::binary);
	valstream.clear();
	valstream.seekg(16+nTrainSamples()*SampleSize(),std::ifstream::beg);
	printf("file contains data of size %dx%dx%d\n",n_samples,n_rows,n_columns);
}

int MnistData::BigtoLittle(int val) {
	int out;
	char* cout = ((char*)&out);
	char* valbytes = reinterpret_cast<char*>(&val);
	cout[0] = valbytes[3];
	cout[1] = valbytes[2];
	cout[2] = valbytes[1];
	cout[3] = valbytes[0];
	return out;
}



shared_ptr<float> MnistData::GetNextBatch(int batchsize) {
	int batchbytes = batchsize * n_rows * n_columns;
	int bufsize = n_rows * n_columns;
	char buf[bufsize];
	shared_ptr<float> data(new float[batchbytes],
			[](float* ptr) {
				delete[] ptr;
			});
	int rd = 0;
	int idata = 0;
	while (batchbytes > 0) {
		rd = min(bufsize, batchbytes);
		datastream.read(buf, rd );
		if (datastream.tellg() >= 16+nTrainSamples()*SampleSize()){
			printf("Reached end of training data.  Restarting at beginning\n");
			datastream.clear();
			datastream.seekg(16,std::ifstream::beg);
			datastream.read(buf, rd * sizeof(char));
		}
		for (int i = 0; i < rd; i++) {
			data.get()[idata++] = ((unsigned char) buf[i]) / 255.0f;
		}
		batchbytes -= rd;
	}
	return data;
}

shared_ptr<float> MnistData::GetNextValidationBatch(int batchsize){
	int batchbytes = batchsize * n_rows * n_columns;
	int bufsize = n_rows * n_columns;
	char buf[bufsize];
	shared_ptr<float> data(new float[batchbytes],
			[](float* ptr) {
				delete[] ptr;
			});
	int rd = 0;
	int idata = 0;
	while (batchbytes > 0) {
		rd = min(bufsize, batchbytes);
		valstream.read(buf, rd );
		if (valstream.eof()){
			printf("Reached end of validation data.  Restarting at beginning\n");
			datastream.clear();
			datastream.seekg(16+nTrainSamples()*SampleSize(),std::ifstream::beg);
			datastream.read(buf, rd * sizeof(char));
		}
		for (int i = 0; i < rd; i++) {
			data.get()[idata++] = ((unsigned char) buf[i]) / 255.0f;
		}
		batchbytes -= rd;
	}
	return data;
}

int MnistData::nTrainSamples(){
	return ((int)n_samples*split_);
}

int MnistData::nValSamples(){
	return ((int)n_samples*(1-split_));
}


int MnistData::SampleSize(){
	return n_rows*n_columns;
}


#endif /* APPS_MNIST_RAW_H_ */
