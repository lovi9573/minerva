/*
 * fpga_util.c
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_
#define MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_


void float2q88(float* src, ht_int16* dest, size_t n ){

	for(size_t i = 0; i < n; i++){
		dest[i] = static_cast<ht_int16>(src[i] * (1<<8));
	}
}


void q882float(ht_int16* src, float* dest, size_t n){
	for (size_t i = 0; i < n; i++){
		dest[i] = (static_cast<float>(src[i])) / (1>>8);
	}
}

#endif /* MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_ */
