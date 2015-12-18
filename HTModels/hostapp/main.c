#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include "../HTModels.h"

void usage (char *);

size_t prod(size_t* s){
	return s[0]*s[1]*s[2]*s[3];
}


void test_fwd(){
	uint16_t FRACW = 0;

	/*
	 * Top diff
	 */
	int16_t top_expected_raw[] = {544};
	size_t top_diff_dims[] = {1,1,1,1};
	size_t top_diff_size = prod(top_diff_dims)*sizeof(int16_t);
	int16_t top_produced_raw[prod(top_diff_dims)];

	/*
	 * Filter
	 */
	int16_t filter_raw[] ={11,12,13,14,15,16,17,18};
	size_t filter_dims[] = {2, 2, 2, 1};
	size_t filter_size = prod(filter_dims)*sizeof(int16_t);

	/*
	 * Bottom
	 */

	int16_t bottom_raw[] = {1,2,3,4,5,6,7,8};
	size_t bottom_dims[] = {2, 2, 2, 1};
	size_t bottom_size = prod(bottom_dims)*sizeof(int16_t);

	/*
	 * other
	 */
	int stride = 1;


	conv_forward(bottom_raw, bottom_dims[3], bottom_dims[0], bottom_dims[2], bottom_size,
			filter_raw, filter_dims[3], filter_dims[1],  stride, filter_size,
			top_produced_raw, top_diff_size,
			FRACW);

    int err_cnt = 0;
    printf("====== Convolution Forward [End]=====\n");
    for (size_t i = 0; i < prod( top_diff_dims); i++) {
		if (top_expected_raw[i] != top_produced_raw[i]) {
			printf("output[%llu] is %" PRId16", should be %"PRId16"\n",
			(long long)i, top_produced_raw[i], top_expected_raw[i]);
			err_cnt++;
		}else{
			//printf("output[%llu] is %llu!!!!\n",(long long)i, (long long)output_q88_data[i]);
		}
	}

    printf("\t#Errors: %d\n",err_cnt);


    if (err_cnt)
	printf("FAILED: detected %d issues!\n", err_cnt);
    else
	printf("PASSED\n");
}


void test_back_data(){
	uint16_t FRACW = 0;

	/*
	 * Top diff
	 */
	int16_t top_diff_raw[] = {1,2,3,4,5,6,7,8};
	size_t top_diff_dims[] = {2,2,2,1};
	size_t top_diff_size = prod(top_diff_dims)*sizeof(int16_t);



	/*
	 * Filter
	 */
	int16_t filter_raw[] = {11,12,13,14,15,16,17,18};
	size_t filter_dims[] = {2, 2, 2, 1};
	size_t filter_size = prod(filter_dims)*sizeof(int16_t);

	/*
	 * Bottom diff
	 */

	int16_t bottom_diff_expected_raw[] = {14,41,26,54,130,74,36,81,44,18,53,34,70,170,98,48,109,60};
	size_t bottom_diff_dims[] = {3, 3, 2, 1};
	size_t bottom_diff_size = prod(bottom_diff_dims)*sizeof(int16_t);
	int16_t bottom_diff_produced_raw[prod(bottom_diff_dims)];



	/*
	 * other
	 */
	int pad_x = 0;
	int pad_y = 0;
	int stride = 1;


    conv_backward_data_ht(top_diff_raw, top_diff_size,
    					  filter_raw, filter_dims[2], filter_dims[0], stride, pad_x, pad_y, filter_size,
						  bottom_diff_produced_raw, bottom_diff_dims[0], bottom_diff_dims[1], bottom_diff_dims[2], bottom_diff_dims[3], bottom_diff_size,
						  FRACW );


    int err_cnt = 0;
    printf("====== Convolution Backward Data [End]=====\n");
    for (size_t i = 0; i < prod(bottom_diff_dims); i++) {
		if (bottom_diff_expected_raw[i] != bottom_diff_produced_raw[i]) {
			printf("output[%llu] is %" PRId16", should be %"PRId16"\n",
			(long long)i, bottom_diff_produced_raw[i], bottom_diff_expected_raw[i]);
			err_cnt++;
		}else{
			//printf("output[%llu] is %llu!!!!\n",(long long)i, (long long)output_q88_data[i]);
		}
	}

    printf("\t#Errors: %d\n",err_cnt);


    if (err_cnt)
	printf("FAILED: detected %d issues!\n", err_cnt);
    else
	printf("PASSED\n");
}

void test_back_bias(){
	 printf("====== Convolution Backward Bias [Start]=====\n");
	uint16_t FRACW = 0;

	/*
	 * Top diff
	 */
	int16_t top_diff_raw[] = {1,2,3,4,5,6,7,8};
	size_t top_diff_dims[] = {2,2,2,1};
	size_t top_diff_size = prod(top_diff_dims)*sizeof(int16_t);

	/*
	 * Bias dims
	 */
	int16_t bias_raw[] = {10,26};
	size_t bias_dims[] = {1,1,2,1};
	size_t bias_size = prod(bias_dims)*sizeof(int16_t);
	int16_t bias_produced_raw[prod(bias_dims)];

	conv_backward_bias_ht(top_diff_raw, top_diff_size, top_diff_dims[0],top_diff_dims[1],
					    top_diff_dims[2],top_diff_dims[3],
						bias_produced_raw, bias_size,
						FRACW);

    int err_cnt = 0;
    printf("====== Convolution Backward Bias [End]=====\n");
    for (size_t i = 0; i < prod(bias_dims); i++) {
		if (bias_raw[i] != bias_produced_raw[i]) {
			printf("output[%llu] is %" PRId16", should be %"PRId16"\n",
			(long long)i, bias_produced_raw[i], bias_raw[i]);
			err_cnt++;
		}else{
			//printf("output[%llu] is %llu!!!!\n",(long long)i, (long long)output_q88_data[i]);
		}
	}

    printf("\t#Errors: %d\n",err_cnt);

    if (err_cnt)
	printf("FAILED: detected %d issues!\n", err_cnt);
    else
	printf("PASSED\n");
}

void test_back_filter(){
	uint16_t FRACW = 0;

	/*
	 * Top diff
	 */
	int16_t top_diff_raw[] = {8,6,4,2};
	size_t top_diff_dims[] = {2,2,1,1};
	size_t top_diff_size = prod(top_diff_dims)*sizeof(int16_t);

	/*
	 * Filter
	 */
	int16_t filter_expected_raw[] = {306,286,246,226};
	size_t filter_dims[] = {2, 2, 1, 1};
	size_t filter_size = prod(filter_dims)*sizeof(int16_t);
	int16_t filter_produced_raw[prod(filter_dims)];

	/*
	 * Bottom
	 */

	int16_t bottom_raw[] = {10,11,12,13,14,15,16,17,18};
	size_t bottom_dims[] = {3, 3, 2, 1};
	size_t bottom_size = prod(bottom_dims)*sizeof(int16_t);

	/*
	 * other
	 */
	int pad_x = 0;
	int pad_y = 0;
	int stride = 1;


    conv_backward_filter_ht(top_diff_raw, top_diff_size, top_diff_dims[0],top_diff_dims[1],top_diff_dims[2],top_diff_dims[3],
    						filter_produced_raw, filter_size, filter_dims[0], stride, pad_x, pad_y,
							bottom_raw, bottom_size, bottom_dims[0],bottom_dims[1],bottom_dims[2],
							FRACW );

    int err_cnt = 0;
    printf("====== Convolution Backward Filter [End]=====\n");
    for (size_t i = 0; i < prod(filter_dims); i++) {
		if (filter_expected_raw[i] != filter_produced_raw[i]) {
			printf("output[%llu] is %" PRId16", should be %"PRId16"\n",
			(long long)i, filter_produced_raw[i], filter_expected_raw[i]);
			err_cnt++;
		}else{
			//printf("output[%llu] is %llu!!!!\n",(long long)i, (long long)output_q88_data[i]);
		}
	}

    printf("\t#Errors: %d\n",err_cnt);


    if (err_cnt)
	printf("FAILED: detected %d issues!\n", err_cnt);
    else
	printf("PASSED\n");
}




// Print usage message and exit with error.
void
usage (char* p)
{
    printf("usage: %s [count (default 100)] \n", p);
    exit (1);
}

int main(int argc, char **argv)
{
	test_fwd();
//	test_back_data();
//	test_back_bias();
//	test_back_filter();
}
