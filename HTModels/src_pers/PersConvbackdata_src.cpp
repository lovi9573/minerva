/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersConvbackdata.h"

#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }
#define UINT16_MASK 0xFFFF


void
CPersConvbackdata::PersConvbackdata()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		//Setup
		case CONVBACKDATA_ENTRY: {
			P_top_idx = 0;
			P_top_channel_idx = 0;
			P_bias_idx = 0;
			HtContinue(CONVBACKDATA_RTN);
		}
		break;
		case CONVBACKDATA_RTN: {
			BUSY_RETRY(SendReturnBusy_conv_back_data());
			SendReturn_conv_back_data();
		}
		break;
		default:
			assert(0);
		}
	}
}
