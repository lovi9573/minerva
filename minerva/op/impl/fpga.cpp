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
	size_t img_bytes = img_numbers*sizeof(element_t);
	size_t filter_numbers = inputs[1].size_.Prod();
	size_t filter_bytes = filter_numbers*sizeof(element_t);
	size_t output_numbers = outputs[0].size_.Prod();
	size_t output_bytes = output_numbers*sizeof(element_t);




	printf("num_img: %d, img_dim: %d, img_channels: %d, num_filters: %d, filter_dim: %d, stride: %d",
			inputs[0].size_[3], inputs[0].size_[1], inputs[0].size_[2],
			inputs[1].size_[3], inputs[1].size_[1], closure.stride_vertical);

	conv_forward(input_data, inputs[0].size_[3], inputs[0].size_[1], inputs[0].size_[2], img_bytes,
				filter_data, inputs[1].size_[3], inputs[1].size_[1], closure.stride_vertical, filter_bytes,
				output_data, output_bytes,
				FIXED_POINT_FRACTION_WIDTH);
}

void ConvBackwardData(const DataList& inputs, const DataList& outputs, ConvBackwardDataClosure& closure){
	CHECK_EQ(inputs.size(), 2) << " Convolution #inputs are wrong.\n";
	CHECK_EQ(outputs.size(), 1) << " Convolution #outputs are wrong.\n";

	element_t* bottom_diff = outputs[0].data_;
	Scale bottom_size = outputs[0].size_;
	int bottom_column_stride = bottom_size[0];
	int bottom_channel_stride = bottom_size[1]*bottom_size[0];
	int bottom_image_stride = bottom_size[2]*bottom_size[1]*bottom_size[0];

	element_t* filters = inputs[1].data_;
	Scale filtersize = inputs[1].size_;
	int filter_column_stride = filtersize[0];
	int filter_channel_stride = filtersize[1]*filtersize[0];
	int filter_element_stride = filtersize[2]*filtersize[1]*filtersize[0];

	//element_t* bias = inputs[2].data_;

	element_t* top_diff = inputs[0].data_;
	Scale top_size = inputs[0].size_;
	//int top_column_stride = top_size[0];
	//int top_channel_stride = top_size[1]*top_size[0];
	//int top_image_stride = top_size[2]*top_size[1]*top_size[0];

	for(int i = 0; i < bottom_size.Prod(); i++){
		bottom_diff[i] = 0.0f;
	}


    int pad_height = closure.pad_height;
    int pad_width = closure.pad_width;
    int stride_vertical = closure.stride_vertical;
    int stride_horizontal = closure.stride_horizontal;

    /*
    printf("bottom: (%d,%d) padding: (%d,%d), filter: (%d,%d), top: (%d,%d)\n",
    		bottom_size[0], bottom_size[1],
			pad_width, pad_height,
			filtersize[0], filtersize[1],
			top_size[0], top_size[1]);
*/
    conv_backward_data_ht(top_diff, top_size,
    						filters, filtersize[3], filtersize[1], closure.stride_vertical, filtersize,
							bottom_diff, bottom_size[0], bottom_size[1], bottom_size[2], bottom_size[3], bottom_size,
							FIXED_POINT_FRACTION_WIDTH)
}

/*
 * NOTE: This implementation is not designed to be fast.  If you want that, use a GPU.  It is meant to provide feature completeness and the ability to develop on a GPU-less machine.
 */
void ConvBackwardBias(const DataList& inputs, const DataList& outputs, ConvBackwardBiasClosure& closure){
	CHECK_EQ(inputs.size(), 1) << " Convolution #inputs are wrong.\n";
	CHECK_EQ(outputs.size(), 1) << " Convolution #outputs are wrong.\n";

	element_t* bottom_diff = outputs[0].data_;
	Scale bottom_size = outputs[0].size_;
	//int bottom_image_stride = bottom_size[0];
	//cout << "convolution bias_diff size " << bottom_size.ToString() << "\n";

	element_t* top_diff = inputs[0].data_;
	Scale top_size = inputs[0].size_;
	int top_column_stride = top_size[0];
	int top_channel_stride = top_size[1]*top_size[0];
	int top_image_stride = top_size[2]*top_size[1]*top_size[0];


	for(int i = 0; i < top_size[2]; i++){
		bottom_diff[i] = 0.0f;
	}

	//TODO: write for fpga
}

/*
 * NOTE: This implementation is not designed to be fast.  If you want that, use a GPU.  It is meant to provide feature completeness and the ability to develop on a GPU-less machine.
 */
void ConvBackwardFilter(const DataList& inputs, const DataList& outputs, ConvBackwardFilterClosure& closure){
	CHECK_EQ(inputs.size(), 2) << " Convolution #inputs are wrong.\n";
	CHECK_EQ(outputs.size(), 1) << " Convolution #outputs are wrong.\n";

	element_t* bottom = inputs[1].data_;
	Scale bottom_size = inputs[1].size_;
	int bottom_column_stride = bottom_size[0];
	int bottom_channel_stride = bottom_size[1]*bottom_size[0];
	int bottom_image_stride = bottom_size[2]*bottom_size[1]*bottom_size[0];

	element_t* filter_diff = outputs[0].data_;
	Scale filtersize = outputs[0].size_;
	int filter_column_stride = filtersize[0];
	int filter_channel_stride = filtersize[1]*filtersize[0];
	int filter_element_stride = filtersize[2]*filtersize[1]*filtersize[0];

	//element_t* bias = inputs[2].data_;

	element_t* top_diff = inputs[0].data_;
	Scale top_size = inputs[0].size_;
	//int top_column_stride = top_size[0];
	//int top_channel_stride = top_size[1]*top_size[0];
	//int top_image_stride = top_size[2]*top_size[1]*top_size[0];


	for(int i = 0; i < filtersize.Prod(); i++){
		filter_diff[i] = 0.0f;
	}


    int pad_height = closure.pad_height;
    int pad_width = closure.pad_width;
    int stride_vertical = closure.stride_vertical;
    int stride_horizontal = closure.stride_horizontal;

    int topindex = -1  ;

    //TODO: write for fpga
}




} // namespace basic
}// namespace minerva

#endif
