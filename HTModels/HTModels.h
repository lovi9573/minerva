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

void conv_backward_bias_ht(void* top_diff, size_t top_elements, size_t top_dim_x, size_t top_dim_y,
						   size_t top_n_channels, size_t top_n_samples,
						void* bottom_diff, size_t bottom_alloc,
						int frac_w);

void conv_backward_filter_ht(void* top_diff, size_t top_size, size_t top_dim_x, size_t top_dim_y, size_t top_dim_c, size_t top_dim_n,
							 void* filter_diff, size_t filter_size, size_t filter_dim_x, size_t filter_column_stride, size_t pad_x, size_t pad_y,
							 void* bottom, size_t bottom_size, size_t bottom_dim_x, size_t bottom_dim_y, size_t bottom_dim_c,
							 int frac_width);


#endif /* HTMODELS_HTMODELS_H_ */
