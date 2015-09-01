/*
 * fpga.c
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */


#ifdef HAS_FPGA

#include "fpga.h"
#include "op/impl/fpga/fpga_util.h"
#include "op/closure.h"
#include "../HTModels/HTModels.h"


namespace minerva {
namespace fpga {



void ReluForward(const DataList& inputs, const DataList& outputs, ReluForwardClosure& c) {
  CHECK_EQ(inputs.size(), 1) << "relu forward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "relu forward #outputs wrong";

  element_t* input_data = inputs[0].data_;
  element_t* output_data = outputs[0].data_;
  size_t numbers = inputs[0].size_.Prod();
  size_t bytes = numbers*2;
  //Ensure that the allocation can be interpreted as uint64_t
  if (bytes % 8 != 0){
	  bytes = (bytes/8+1)*8;
  }

  char *input_q88_data = (char *)malloc(bytes);
  char *output_q88_data = (char *)malloc(bytes);

  element_t2qxx(input_data, input_q88_data,numbers,16,8);

  //relu_forward(input_q88_data, output_q88_data, numbers );

	qxx2element_t(output_q88_data, output_data, numbers,8,8);
	free(input_q88_data);
	free(output_q88_data);

}

void ConvForward(const DataList& inputs, const DataList& outputs, ConvForwardClosure& closure){
	element_t* input_data = inputs[0].data_;
	element_t* filter_data = inputs[1].data_;
	element_t* output_data = outputs[0].data_;
	size_t img_numbers = inputs[0].size_.Prod();
	size_t img_bytes = img_numbers*2;
	size_t filter_numbers = inputs[1].size_.Prod();
	size_t filter_bytes = filter_numbers*2;
	size_t output_numbers = outputs[0].size_.Prod();
	size_t output_bytes = output_numbers*2;
	int frac_width = 0;


	//Ensure that the allocation can be interpreted as uint64_t
	if (img_bytes % 8 != 0){
		img_bytes = (img_bytes/8+1)*8;
	}
	if (filter_bytes % 8 != 0){
		filter_bytes = (filter_bytes/8+1)*8;
	}
	if (output_bytes % 8 != 0){
		output_bytes = (output_bytes/8+1)*8;
	}

	char *input_q88_data = (char *)malloc(img_bytes);
	char *filter_q88_data = (char *)malloc(filter_bytes);
	char *output_q88_data = (char *)malloc(output_bytes);

	element_t2qxx(input_data, input_q88_data,img_numbers,16,FIXED_POINT_FRACTION_WIDTH);
	element_t2qxx(filter_data, filter_q88_data,filter_numbers,16,FIXED_POINT_FRACTION_WIDTH);
	/*
	  closure.pad_height;
	  int pad_width;
	  closure.stride_vertical;
	  int stride_horizontal;

	conv_forward(input_q88_data, size_t num_img, size_t img_dim, size_t img_channels, size_t img_alloc,
			 void *filters_q88_data, size_t num_filters, closure.pad_height, closure.stride_vertical, size_t filter_alloc,
			 void *output_q88_data, size_t output_alloc,
			 uint16_t fraction_width);
	*/
	printf("num_img: %d, img_dim: %d, img_channels: %d, num_filters: %d, filter_dim: %d, stride: %d",
			inputs[0].size_[3], inputs[0].size_[1], inputs[0].size_[2],
			inputs[1].size_[3], inputs[1].size_[1], closure.stride_vertical);

	conv_forward(input_q88_data, inputs[0].size_[3], inputs[0].size_[1], inputs[0].size_[2], img_bytes,
				 filter_q88_data, inputs[1].size_[3], inputs[1].size_[1], closure.stride_vertical, filter_bytes,
				 output_q88_data, output_bytes,
				 frac_width);

	qxx2element_t(output_q88_data, output_data, output_numbers,16,FIXED_POINT_FRACTION_WIDTH);
	for(size_t i = 0; i < output_numbers; i++){
		printf("%f\n",output_data[i]);
	}
	free(input_q88_data);
	free(filter_q88_data);
	//free(output_q88_data);
}



} // namespace basic
}// namespace minerva

#endif
