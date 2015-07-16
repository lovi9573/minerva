/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersCtl.h"


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


