/*
 * fpga.h
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_OP_IMPL_FPGA_H_
#define MINERVA_OP_IMPL_FPGA_H_

#ifdef HAS_FPGA

#include "../physical_fn.h"
#include "op/closure.h"

namespace minerva {
namespace fpga {

void ReluForward(const DataList&, const DataList&, ReluForwardClosure&);
void ConvForward(const DataList&, const DataList&, ConvForwardClosure&);

} // namespace fpga
}// namespace minerva

#endif

#endif /* MINERVA_OP_IMPL_FPGA_H_ */
