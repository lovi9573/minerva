/*
 * fpga_util.c
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_
#define MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_


void float2qxx(float* src, int64_t* dest, size_t n , int integerdigits, int fractionaldigits){

	for(size_t i = 0; i < n; i++){
		dest[i] = static_cast<uint64_t>(src[i]); // * (1<<fractionaldigits));
		printf("float2q%d.%d %f => %ld\n",integerdigits, fractionaldigits, src[i],dest[i]);
	}
}


void qxx2float(int64_t* src, float* dest, size_t n, int integerdigits, int fractionaldigits){
	for (size_t i = 0; i < n; i++){
		dest[i] = (static_cast<float>(src[i])); // / (1>>fractionaldigits);
		printf("q%d.%d_2float %ld => %f\n",integerdigits, fractionaldigits, src[i],dest[i]);
	}
}

#endif /* MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_ */
