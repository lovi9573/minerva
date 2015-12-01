/*
 * mnist_raw.h
 *
 *  Created on: Oct 14, 2015
 *      Author: jlovitt
 */

#ifndef APPS_DATA_PROVIDER_H_
#define APPS_DATA_PROVIDER_H_


#include <cstdio>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <memory>

using namespace std;

class DataProvider{
public:
	/**
	 * Return a pointer to a batch of training data.
	 * @param batchsize The number of samples to return.
	 */
	virtual shared_ptr<float> next_batch(int batchsize) = 0;

	/**
	 * Return a pointer to a batch of validation data.
	 * @param batchsize The number of samples to return.
	 */
	virtual shared_ptr<float> next_val_batch(int batchsize) = 0;

	virtual int n_train_samples() = 0;
	virtual int n_val_samples() = 0;

	/**
	 * Returns the number of channels in a training sample
	 */
	virtual int n_channels() = 0;

	/*
	 * Returns the size of a training sample in the y direction
	 */
	virtual int dim_y() = 0;

	/*
	 * Returns the size of a training sample in the x direction
	 */
	virtual int dim_x() = 0;

	/*
	 * Returns the size of a training sample as a number of float32 values.
	 */
	virtual int sample_size() = 0;
};



#endif /* APPS_DATA_PROVIDER_H_ */
