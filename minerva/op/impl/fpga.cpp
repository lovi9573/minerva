/*
 * fpga.c
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */


#ifdef HAS_FPGA

#include "fpga.h"
#include "op/impl/fpga/fpga_util.h"
#include "op/closure.h"
#include "../HTModels/HTModels.h"


namespace minerva {
namespace fpga {



void ReluForward(const DataList& inputs, const DataList& outputs, ReluForwardClosure& c) {
  CHECK_EQ(inputs.size(), 1) << "relu forward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "relu forward #outputs wrong";

  float* input_data = inputs[0].data_;
  float* output_data = outputs[0].data_;
  size_t numbers = inputs[0].size_.Prod();
  size_t bytes = numbers*2;

  char *input_q88_data = (char *)malloc(bytes);
  char *output_q88_data = (char *)malloc(bytes);

  float2qxx(input_data, input_q88_data,numbers,16,8);

  //relu_forward(input_q88_data, output_q88_data, numbers );

/*
 * HT interface
 *
	// Get interface from context
  	int unitCnt = ctx.pHt_host_interface->GetUnitCnt();

	// Set up memory
	float* ht_input_data = (ht_int16*)ctx.pHt_host_interface->MemAlloc(numbers * sizeof(ht_int16));;
	float* ht_output_data = (ht_int16*)ctx.pHt_host_interface->MemAlloc(numbers * sizeof(ht_int16));
	if(!ht_intput_data){
		LOG(FATAL) << "HT MemAlloc failed.";
	}
	ctx.pHt_host_interface->MemCpy(ht_input_data, input_q88_data, numbers * sizeof(ht_int16));

	// avoid bank aliasing for performance
	if (unitCnt > 16 && !(unitCnt & 1)) unitCnt -= 1;
	printf("stride = %d\n", unitCnt);
	fflush(stdout);

	// Initialize modules with messages
	ctx.pHt_host_interface->SendAllHostMsg(IN_ADDR, (uint64_t)ht_input_data);
	ctx.pHt_host_interface->SendAllHostMsg(OUT_ADDR, (uint64_t)ht_output_data);
	ctx.pHt_host_interface->SendAllHostMsg(VEC_LEN, (uint64_t)numbers);

	// Send calls to units
	for (int unit = 0; unit < unitCnt; unit++)
		//									offset, stride
		ctx.pAuUnits[unit]->SendCall_htmain(unit , unitCnt );

	// Wait for returns
	for (int unit = 0; unit < unitCnt; unit++) {
		while (!ctx.pAuUnits[unit]->RecvReturn_htmain())
			usleep(1000);
	}

	//Copy results out.
	ctx.pHt_host_interface->MemCpy(output_q88_data, ht_output_data, numbers * sizeof(ht_int16));
*
 * End HT Interface
 */

	qxx2float(output_q88_data, output_data, numbers,8,8);
	free(input_q88_data);
	free(output_q88_data);

}



} // namespace basic
}// namespace minerva

#endif
