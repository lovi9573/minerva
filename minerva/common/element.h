/*
 * element.h
 *
 *  Created on: Jul 27, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_COMMON_ELEMENT_H_
#define MINERVA_COMMON_ELEMENT_H_

#include "common/fixedpoint.h"


#if defined(FIXED_POINT) || defined(HAS_FPGA)
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE, FIXED_POINT_TYPE, FIXED_POINT_FRACTION_WIDTH_N> element_t;
#else
	typedef float element_t;
#endif


#endif /* MINERVA_COMMON_ELEMENT_H_ */
