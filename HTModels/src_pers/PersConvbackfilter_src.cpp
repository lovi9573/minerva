/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersConvbackfilter.h"

#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }
#define UINT16_MASK 0xFFFF


void
CPersConvbackfilter::PersConvbackfilter()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		//Setup
		case CONVBACKFILTER_ENTRY: {
			HtContinue(CONVBACKFILTER_RTN);
		}
		break;
		case CONVBACKFILTER_RTN: {
			BUSY_RETRY(SendReturnBusy_conv_back_filter());
			SendReturn_conv_back_filter();
		}
		break;
		default:
			assert(0);
		}
	}
}
