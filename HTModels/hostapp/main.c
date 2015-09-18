#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include "../HTModels.h"
/*

extern void conv_forward(void *input_q88_data, size_t num_img, size_t img_dim, size_t img_channels, size_t img_alloc,
						 void *filters_q88_data, size_t num_filters, size_t filter_dim, size_t stride, size_t filter_alloc,
						 void *output_q88_data, size_t output_alloc,
						 uint16_t fraction_width);

extern void conv_backward_data_ht(void* top_diff, size_t top_alloc,
		void* filter_data, size_t num_filters, size_t filter_dim, size_t stride, size_t filter_alloc,
		void* bottom_diff, int bottom_width, int bottom_height, int bottom_channels, int bottom_samples, size_t bottom_alloc,
		int frac_w );

extern void ConvBackwardBias_ht(void* top_diff, size_t top_alloc, size_t top_column_stride, size_t top_channel_stride, size_t top_image_stride,
		void* bottom_diff, size_t bottom_alloc, int channels,
		int frac_w);
*/
void usage (char *);

int main(int argc, char **argv)
{
	uint16_t FRACW = 8;

	/*
	 * Bottom dims
	 */
	size_t num_img =1;
	size_t img_dim = 17;
	size_t img_channels = 2;
	size_t img_size = num_img*img_dim*img_dim*img_channels;
	size_t img_alloc = img_size;
	if(img_alloc%4 != 0){
		img_alloc = (img_alloc/4+1)*4;
	}
	int16_t input_q88_data[img_alloc];

	/*
	 * Filter dims
	 */

	size_t num_filters = 2;
	size_t filter_dim = 3;
	size_t stride = 2;
	size_t filter_size = num_filters*filter_dim*filter_dim*img_channels;
	size_t filter_alloc = filter_size;
	if(filter_alloc%4 != 0){
		filter_alloc = ((filter_alloc+3)/4)*4;
	}

	/*
	 * Bias dims
	 */
	size_t bias_size = num_filters;
	size_t bias_alloc = bias_size;
	if(bias_alloc%4 != 0){
		bias_alloc = ((bias_alloc+3)/4)*4;
	}


	/*
	 * Top dims
	 */
	size_t num_samples = num_img;
	size_t top_dim = ((img_dim-filter_dim)/stride+1);
	size_t top_channels = num_filters;
	size_t top_size = num_samples*top_dim*top_dim*top_channels;
	size_t top_alloc = top_size;
	if(top_alloc%4 != 0){
		top_alloc = ((top_alloc+3)/4 )*4;
	}


	int16_t filters_q88_data[filter_alloc];
	int16_t bias_data[bias_alloc];
	int16_t top_diff_data[top_alloc];

    int err_cnt = 0;

    int16_t expected_output = (int16_t)((filter_dim*filter_dim*img_channels));

	size_t output_elements = num_samples*top_dim*top_dim*top_channels;
	if(output_elements%4 != 0){
		printf("Implementation requires that\n"
				"\tnum_img*((img_dim-filter_dim)/stride+1)*((img_dim-filter_dim)/stride+1)*num_filters\n"
				"be a multiple of 4.\n"
				"got %lu\n",output_elements);
		exit(1);
	}
	size_t output_alloc = output_elements;
	if(output_alloc%4 != 0){
		output_alloc = ((output_alloc+3)/4)*4;
	}
	int16_t output_q88_data[output_alloc];


    for (size_t i = 0; i < img_size; i++) {
    	input_q88_data[i] = 1 << (FRACW-1);
    }


    for (size_t i = 0; i < filter_size; i++) {
    	filters_q88_data[i] = (1 << (FRACW-1));
    }

/*
    conv_forward(input_q88_data,  num_img,  img_dim,  img_channels, img_alloc,
    			 filters_q88_data,  num_filters,  filter_dim,  stride, filter_alloc,
    			 output_q88_data, output_alloc,
				 FRACW);

    printf("Conv done\n");

    // check results
    if(expected_output > (1<< (16 - (FRACW-2)))){
    	printf("Expected Cieling\n");
    	expected_output = 0x7FFF;
    	//expected_output = 0x8000;
    }else{
    	expected_output = expected_output << (FRACW-2);
    }
    printf("====== Convolution Forward =====\n");
    for (size_t i = 0; i < output_elements; i++) {
			if (output_q88_data[i] != expected_output) {
				printf("output[%llu] is %hx, should be %hx\n",
				(long long)i, output_q88_data[i], expected_output);
				err_cnt++;
			}else{
				//printf("output[%llu] is %llu!!!!\n",(long long)i, (long long)output_q88_data[i]);
			}
    }
*/

	/*
	 * Backward bias
	 */
    printf("====== Convolution Backward Bias =====\n");
    for(int i =0; i < (int)top_alloc; i++){
    	top_diff_data[i] = 1;
    }
    expected_output = num_samples*top_dim*top_dim;

    ConvBackwardBias_ht(top_diff_data, top_alloc,	top_dim, top_dim*top_dim, top_dim*top_dim*top_channels,
    					bias_data, bias_alloc, bias_size,
						FRACW);
    for (size_t i = 0; i < bias_size; i++) {
		if (bias_data[i] != expected_output) {
			printf("output[%llu] is %hx, should be %hx\n",
			(long long)i, bias_data[i], expected_output);
			err_cnt++;
		}else{
			//printf("output[%llu] is %llu!!!!\n",(long long)i, (long long)output_q88_data[i]);
		}
	}



    if (err_cnt)
	printf("FAILED: detected %d issues!\n", err_cnt);
    else
	printf("PASSED\n");

    return err_cnt;
}

// Print usage message and exit with error.
void
usage (char* p)
{
    printf("usage: %s [count (default 100)] \n", p);
    exit (1);
}

