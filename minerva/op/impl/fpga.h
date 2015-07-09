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
namespace basic {

void ReluForward(const DataList&, const DataList&, ReluForwardClosure&);

} // namespace basic
}// namespace minerva

#endif

#endif /* MINERVA_OP_IMPL_FPGA_H_ */
