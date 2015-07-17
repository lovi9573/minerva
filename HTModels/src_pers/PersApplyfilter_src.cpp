/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersApplyfilter.h"

#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }



void
CPersApplyfilter::PersApplyfilter()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case RELU_LD_FILTER: {
			BUSY_RETRY(ReadMemBusy());
			printf("filter");
			ReadMem_filter(SR_filterAddr,0, 9);
			ReadMemPause(RELU_LD_IMG_PATCH);
		}
		break;
		case RELU_LD_IMG_PATCH: {
			BUSY_RETRY(WriteMemBusy());
			HtContinue(RELU_RTN);
		}
		break;
		case RELU_APPLY: {
			BUSY_RETRY(SendReturnBusy_applyfilter());

			SendReturn_applyfilter();
		}
		break;
		case RELU_WRITE: {
			BUSY_RETRY(SendReturnBusy_applyfilter());

			SendReturn_applyfilter();
		}
		break;
		case RELU_RTN: {
			BUSY_RETRY(SendReturnBusy_applyfilter());

			SendReturn_applyfilter();
		}
		break;
		default:
			assert(0);
		}
	}
}
