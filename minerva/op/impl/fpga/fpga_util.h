/*
 * fpga_util.c
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_
#define MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_


//TODO: does not take machine endianness into account.
void float2qxx(float* src, char* dest, size_t n , int width, int fractionaldigits){
	int32_t maxint = 1<<(width - fractionaldigits -1);

	for(size_t i = 0; i < n; i++){
		if(src[i] >= maxint){
			reinterpret_cast<int16_t*>(dest)[i] = 0x7FFF;
		}
		else if(src[i] <= -maxint){
			reinterpret_cast<int16_t*>(dest)[i] = 0x1000;
		}
		else{
			reinterpret_cast<int16_t*>(dest)[i] = static_cast<int16_t>(src[i]* (1<<fractionaldigits));
		}
		printf("float2q%d.%d %f => %d / (2^%d)\n",width, fractionaldigits, src[i],reinterpret_cast<int16_t*>(dest)[i],fractionaldigits);
	}
}


void qxx2float(char* src, float* dest, size_t n, int width, int fractionaldigits){
	for (size_t i = 0; i < n; i++){
		int16_t intval = reinterpret_cast<int16_t*>(src)[i];
		dest[i] = static_cast<float>(intval)/ (1<<fractionaldigits);
		printf("q%d.%d_2float %d /%d => %f\n",width, fractionaldigits, reinterpret_cast<int16_t*>(src)[i], (1 << fractionaldigits),dest[i]);
	}
}


/*

void float2qxx(float* src, char* dest, size_t n , int width, int fractionaldigits){
	int integerdigits = width-fractionaldigits-1;
	unsigned char signmask = 0x80;
	unsigned char intmask = ((unsigned char)0xFF << (8 - integerdigits)) >> 1;
	unsigned char fracmask = ~(signmask | intmask);


	for(size_t i = 0; i < n; i++){
		int sign = (src[i] > 0)? 0 : 1;
		int integer = static_cast<char>(src[i]);
		int fractional = static_cast<int>(src[i]* (1<<fractionaldigits));
		dest[i*2] = (sign * signmask) | (static_cast<char>(integer << (8 - integerdigits-1)) & intmask) |(fractional >> 8 & fracmask) ;
		dest[i*2+1] = static_cast<char>(fractional & 0xFF) ;
		printf("float2q%d.%d %f => %x %x\n",width, fractionaldigits, src[i],dest[i*2],dest[i*2+1]);
	}
}


void qxx2float(char* src, float* dest, size_t n, int width, int fractionaldigits){
	for (size_t i = 0; i < n; i++){
		dest[i] = (static_cast<float>(src[i*2])/ (1<<fractionaldigits) + static_cast<float>(src[i*2+1])); // / (1>>fractionaldigits);
		printf("q%d.%d_2float %d %d/256 => %f\n",width, fractionaldigits, src[i*2+1],src[i*2],dest[i]);
	}
}
*/

#endif /* MINERVA_OP_IMPL_FPGA_FPGA_UTIL_C_ */
