/*
 * element.h
 *
 *  Created on: Jul 27, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_COMMON_ELEMENT_H_
#define MINERVA_COMMON_ELEMENT_H_


#if defined(FIXED_POINT) || defined(HAS_FPGA)
	typedef element_t  int16_t;
#else
	typedef element_t float	;
#endif

#if defined(FIXED_POINT) || defined(HAS_FPGA)
	template<typename w>
	element_t operator*(element_t rhs){
		return
	}
#else
	float
#endif



#endif /* MINERVA_COMMON_ELEMENT_H_ */
