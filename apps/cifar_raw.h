/*
 * mnist_raw.h
 *
 *  Created on: Oct 14, 2015
 *      Author: jlovitt
 */

#ifndef APPS_CIFAR_RAW_H_
#define APPS_CIFAR_RAW_H_


#include <cstdio>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <memory>

using namespace std;

class CifarData{
public:
	CifarData(char* data_filename,float);
	shared_ptr<float> GetNextBatch(int batchsize);
	shared_ptr<float> GetNextValidationBatch(int batchsize);
	void set_split(float);
	int n_train_samples();
	int n_channels();
	int dim_y();
	int dim_x();
	int n_val_samples();
	int sample_size();
private:
	int n_samples;
	int n_rows;
	int n_columns;
	int n_channels_;
	float split_;
	int trainpos_;
	ifstream datastream;
	ifstream valstream;
};


CifarData::CifarData(char* data_filename, float split) {
	printf("opening \"%s\"\n",data_filename);
	datastream.open(data_filename, std::ifstream::binary);
	datastream.clear();
	datastream.seekg(0,std::ifstream::beg);
	n_samples = 10000;
	n_rows = 32;
	n_columns = 32;
	n_channels_=3;
	split_ = split;
	valstream.open(data_filename, std::ifstream::binary);
	valstream.clear();
	valstream.seekg(n_train_samples()*(sample_size()+1),std::ifstream::beg);
	printf("file contains data of size %dx%dx%dx%d\n",n_samples,n_channels_,n_rows,n_columns);
}

/**
 * Returns a batch of cifar image data in row major order where color channels are separated.
 * i.e. since an image is 32x32 pixels, 1024 red data, will be followed by 1024 green data, etc...
 */
shared_ptr<float> CifarData::GetNextBatch(int batchsize) {
	int batchbytes = batchsize *(n_channels_* n_rows * n_columns + 1);
	int bufsize = n_channels_*n_rows * n_columns+1;
	char buf[bufsize];
	shared_ptr<float> data(new float[batchbytes - batchsize],
			[](float* ptr) {
				delete[] ptr;
			});
	int rd = 0;
	int idata = 0;
	while (batchbytes > 0) {
		rd = min(bufsize, batchbytes);
		datastream.read(buf, rd  );
		if (datastream.tellg() >= n_train_samples()*(sample_size()+1)){
			printf("Reached end of training data.  Restarting at beginning\n");
			datastream.clear();
			datastream.seekg(0,std::ifstream::beg);
			datastream.read(buf, rd * sizeof(char));
		}
		for (int i = 1; i < rd; i++) {
			data.get()[idata++] = ((unsigned char) buf[i]) / 255.0f;
		}
		batchbytes -= rd ;
	}
	return data;
}

shared_ptr<float> CifarData::GetNextValidationBatch(int batchsize){
	int batchbytes = batchsize *(n_channels_* n_rows * n_columns + 1);
	int bufsize = n_channels_*n_rows * n_columns+1;
	char buf[bufsize];
	shared_ptr<float> data(new float[batchbytes - batchsize],
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
			datastream.seekg(n_train_samples()*(sample_size()+1),std::ifstream::beg);
			datastream.read(buf, rd * sizeof(char));
		}
		for (int i = 1; i < rd; i++) {
			data.get()[idata++] = ((unsigned char) buf[i]) / 255.0f;
		}
		batchbytes -= rd;
	}
	return data;
}

int CifarData::n_train_samples(){
	return ((int)n_samples*split_);
}

int CifarData::n_val_samples(){
	return ((int)n_samples*(1-split_));
}


int CifarData::n_channels(){
	return n_channels_;
}

int CifarData::dim_y(){
	return n_rows;
}
int CifarData::dim_x(){
	return n_columns;
}

int CifarData::sample_size(){
	return n_rows*n_columns*n_channels_;
}



#endif /* APPS_CIFAR_RAW_H_ */
