/*
 * fpga_util.c
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_
#define MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_


void float2qxx(float* src, char* dest, size_t n , int width, int fractionaldigits){

	for(size_t i = 0; i < n; i++){
		dest[i*2] = static_cast<char>(src[i]* (1<<fractionaldigits));
		dest[i*2+1] = static_cast<char>(src[i]); // * (1<<fractionaldigits));
		printf("float2q%d.%d %f => %d %d/256\n",width, fractionaldigits, src[i],dest[i*2+1],dest[i*2]);
	}
}


void qxx2float(char* src, float* dest, size_t n, int width, int fractionaldigits){
	for (size_t i = 0; i < n; i++){
		dest[i] = (static_cast<float>(src[i*2])/ (1<<fractionaldigits) + static_cast<float>(src[i*2+1])); // / (1>>fractionaldigits);
		printf("q%d.%d_2float %d %d/256 => %f\n",width, fractionaldigits, src[i*2+1],src[i*2],dest[i]);
	}
}

#endif /* MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_ */
