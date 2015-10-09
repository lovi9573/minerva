/*
 * HTModels.h
 *
 *  Created on: Jul 13, 2015
 *      Author: jlovitt
 */

#ifndef HTMODELS_HTMODELS_H_
#define HTMODELS_HTMODELS_H_

//#include <Ht.h>


#define CONV_FORWARD 1
#define CONV_BACKWARD_DATA 2
#define CONV_BACKWARD_BIAS 3
#define CONV_BACKWARD_FILTER 4

//void relu_forward(char *input_q88_data, char *output_q88_data, size_t numbers );
void conv_forward(void *input_q88_data, size_t num_img, size_t img_dim, size_t img_channels, size_t img_alloc,
		 void *filters_q88_data, size_t num_filters, size_t filter_dim, size_t stride, size_t filter_alloc,
		 void *output_q88_data, size_t output_alloc,
		 uint16_t fraction_width);

void conv_backward_data_ht(void* top_diff, size_t top_alloc,
							void* filter_data, int num_filters, int filter_dim, int stride, int pad_x, int pad_y, size_t filter_alloc,
							void* bottom_diff, int bottom_width, int bottom_height, int bottom_channels, int bottom_samples, size_t bottom_alloc,
							int frac_w );

void ConvBackwardBias_ht(void* top_diff, size_t top_alloc, size_t top_column_stride, size_t top_channel_stride, size_t top_image_stride,
		void* bottom_diff, size_t bottom_alloc, int channels,
		int frac_w);


#endif /* HTMODELS_HTMODELS_H_ */
