/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersCtl.h"
#include "PersRelu.h"

#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }


void
CPersCtl::PersCtl()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case CTL_ENTRY: {
			HtContinue(CTL_COMPUTE);
		}
		break;
		case CTL_COMPUTE: {
			BUSY_RETRY(SendCallBusy_relu());

			if (P_vecIdx < SR_vecLen) {
				SendCallFork_relu(CTL_JOIN, P_vecIdx);
				HtContinue(CTL_COMPUTE);
				P_vecIdx += P_vecStride;
			} else {
				RecvReturnPause_relu(CTL_RTN);
			}
		}
		break;
		case CTL_JOIN: {
			RecvReturnJoin_relu();
		}
		break;
		case CTL_RTN: {
			BUSY_RETRY(SendReturnBusy_htmain());
			SendReturn_htmain();
		}
		break;
		default:
			assert(0);
		}
	}
}


void
CPersRelu::PersRelu()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case RELU_LD1: {
			BUSY_RETRY(ReadMemBusy());

			// Memory read request
			MemAddr_t memRdAddr = SR_op1Addr + (P_vecIdx << 3);
			printf("Preread  op1<%lu>: %ld\n",memRdAddr.to_uint64(),PR_op1);
			ReadMem_op1(memRdAddr);
			printf("Postread op1<%lu>: %ld\n",memRdAddr.to_uint64(),PR_op1);
			HtContinue(RELU_ST);
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
			MemAddr_t memWrAddr = SR_resAddr + (P_vecIdx << 3);
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
