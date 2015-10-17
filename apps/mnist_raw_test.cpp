/*
 * mnist_raw_test.cpp
 *
 *  Created on: Oct 17, 2015
 *      Author: user
 */



#import "mnist_raw.h"

int main(int argc, char**argv){
	//Load the training data
	printf("load data\n");
	int n_samples,sample_size;
	MnistData dp(argv[1]);
	n_samples = dp.nSamples();
	sample_size = dp.SampleSize();
	printf("%d samples of size %d\n",n_samples,sample_size);

	//Get minibatch
	shared_ptr<float> batch = dp.GetNextBatch(10);
	//NArray visible = NArray::MakeNArray({BATCH_SIZE,sample_size}, batch);
}
