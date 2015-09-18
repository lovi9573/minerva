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
			printf("Task: %d\n",PR_task);
			if(PR_task == CONV_FORWARD ){
				BUSY_RETRY(SendCallBusy_conv_fwd());
				SendCall_conv_fwd(CTL_RTN, PR_rank, PR_rankStride);
			}
/*			else if(PR_task == CONV_BACKWARD_DATA ){
				BUSY_RETRY(SendCallBusy_convfwd());
				SendCall_convfwd(PR_rank, PR_rankStride);
			}*/
			else if(PR_task == CONV_BACKWARD_BIAS && PR_rank == 0 ){
				BUSY_RETRY(SendCallBusy_conv_back_bias());
				SendCall_conv_back_bias(CTL_RTN);
			}
/*			else if(PR_task == CONV_BACKWARD_FILTER ){
				BUSY_RETRY(SendCallBusy_convfwd());
				SendCall_convfwd(PR_rank, PR_rankStride);
			}*/
			else{
				HtContinue(CTL_RTN);
			}
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


