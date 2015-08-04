/*
 * HTModels.h
 *
 *  Created on: Jul 13, 2015
 *      Author: jlovitt
 */

#ifndef HTMODELS_HTMODELS_H_
#define HTMODELS_HTMODELS_H_

//#include <Ht.h>

//void relu_forward(char *input_q88_data, char *output_q88_data, size_t numbers );
void conv_forward(void *input_q88_data, size_t num_img, size_t img_dim, size_t img_channels, size_t img_alloc,
		 void *filters_q88_data, size_t num_filters, size_t filter_dim, size_t stride, size_t filter_alloc,
		 void *output_q88_data, size_t output_alloc,
		 uint16_t fraction_width);



#endif /* HTMODELS_HTMODELS_H_ */
