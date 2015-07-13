/*
 * fpga_util.c
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_
#define MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_


void float2qxx(float* src, uint64_t* dest, size_t n , int integerdigits, int fractionaldigits){

	for(size_t i = 0; i < n; i++){
		dest[i] = static_cast<uint64_t>(src[i] * (1<<fractionaldigits));
	}
}


void qxx2float(uint64_t* src, float* dest, size_t n, int integerdigits, int fractionaldigits){
	for (size_t i = 0; i < n; i++){
		dest[i] = (static_cast<float>(src[i])) / (1>>fractionaldigits);
	}
}

#endif /* MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_ */
