#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>

extern void conv_forward(void *input_q88_data, size_t num_img, size_t img_dim, size_t img_channels, size_t img_alloc,
						 void *filters_q88_data, size_t num_filters, size_t filter_dim, size_t stride, size_t filter_alloc,
						 void *output_q88_data, size_t output_alloc,
						 uint16_t fraction_width);
void usage (char *);

int main(int argc, char **argv)
{
	uint16_t FRACW = 14;

	size_t num_img =1;
	size_t img_dim = 17;
	size_t img_channels = 2;
	size_t img_size = num_img*img_dim*img_dim*img_channels;
	size_t img_alloc = img_size;
	if(img_alloc%4 != 0){
		img_alloc = (img_alloc/4+1)*4;
	}
	int16_t input_q88_data[img_alloc];

	size_t num_filters = 2;
	size_t filter_dim = 3;
	size_t stride = 2;
	size_t filter_size = num_filters*filter_dim*filter_dim*img_channels;
	size_t filter_alloc = filter_size;
	if(filter_alloc%4 != 0){
		filter_alloc = (filter_alloc/4+1)*4;
	}
	int16_t filters_q88_data[filter_alloc];

	size_t output_elements = num_img*((img_dim-filter_dim)/stride+1)*((img_dim-filter_dim)/stride+1)*num_filters;
	if(output_elements%4 != 0){
		printf("Implementation requires that\n"
				"\tnum_img*((img_dim-filter_dim)/stride+1)*((img_dim-filter_dim)/stride+1)*num_filters\n"
				"be a multiple of 4.\n"
				"got %lu\n",output_elements);
		exit(1);
	}
	size_t output_alloc = output_elements;
	if(output_alloc%4 != 0){
		output_alloc = (output_alloc/4+1)*4;
	}
	int16_t output_q88_data[output_alloc];


    for (size_t i = 0; i < img_size; i++) {
    	input_q88_data[i] = 1 << (FRACW-1);
    }


    for (size_t i = 0; i < filter_size; i++) {
    	filters_q88_data[i] = (1 << (FRACW-1));
    }


    conv_forward(input_q88_data,  num_img,  img_dim,  img_channels, img_alloc,
    			 filters_q88_data,  num_filters,  filter_dim,  stride, filter_alloc,
    			 output_q88_data, output_alloc,
				 FRACW);
    printf("Conv done\n");

    // check results
    int err_cnt = 0;

    int16_t expected_output = (int16_t)((filter_dim*filter_dim*img_channels));
    if(expected_output > (1<< (16 - (FRACW-2)))){
    	printf("Expected Cieling\n");
    	expected_output = 0x7FFF;
    	//expected_output = 0x8000;
    }else{
    	expected_output = expected_output << (FRACW-2);
    }
    for (size_t i = 0; i < output_elements; i++) {
			if (output_q88_data[i] != expected_output) {
				printf("output[%llu] is %hx, should be %hx\n",
				(long long)i, output_q88_data[i], expected_output);
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

