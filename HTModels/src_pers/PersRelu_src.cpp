/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersRelu.h"

#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }



void
CPersRelu::PersRelu()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case RELU_LD1: {
			BUSY_RETRY(ReadMemBusy());

			// Memory read request
			//printf("calculate address  idx: %u", P_vecIdx);
			fflush(stdout);
			MemAddr_t memRdAddr = SR_op1Addr + (P_vecIdx << 2);
			//printf("About to read ");
			ReadMem_op1(memRdAddr);
			ReadMemPause(RELU_ST);
		}
		break;
		case RELU_ST: {
			BUSY_RETRY(WriteMemBusy());
			if(PR_op1 > 0){
				P_result = PR_op1;
			}else{
				P_result = 0;
			}
			printf("ST op1: %ld => %ld\n",PR_op1, P_result);

			// Memory write request
			MemAddr_t memWrAddr = SR_resAddr + (P_vecIdx << 2	);
			WriteMem(memWrAddr, P_result);
			WriteMemPause(RELU_RTN);
		}
		break;
		case RELU_RTN: {
			BUSY_RETRY(SendReturnBusy_relu());

			SendReturn_relu();
		}
		break;
		default:
			assert(0);
		}
	}
}
