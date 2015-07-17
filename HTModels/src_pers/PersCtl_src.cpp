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
			printf("Control Entry\n");
			P_applicationIdx_X=(P_rank*SR_stride)%SR_img_dim;
			P_applicationIdx_Y=((P_rank*SR_stride)/SR_img_dim*SR_stride)%SR_img_dim;
			P_applicationIdx_F=(((P_rank*SR_stride)/SR_img_dim*SR_stride)/SR_img_dim)%SR_filter_num;
			HtContinue(CTL_COMPUTE);
		}
		break;
		case CTL_COMPUTE: {
			BUSY_RETRY(SendCallBusy_applyfilter());
			printf("Compute Entry\n");
			if(P_applicationIdx_F  < SR_filter_num){
				if(P_applicationIdx_Y + SR_filter_dim < SR_img_dim){
					if (P_applicationIdx_X <  SR_img_dim - SR_filter_dim) {
						printf("pre-fork");
						SendCallFork_applyfilter(CTL_JOIN, P_applicationIdx_F, P_applicationIdx_X, P_applicationIdx_Y);
						HtContinue(CTL_COMPUTE);
						P_applicationIdx_X += P_applicationStride*SR_stride;
					}else{
						P_applicationIdx_Y += P_applicationIdx_X/(SR_img_dim - SR_filter_dim)*SR_stride;
						P_applicationIdx_X += P_applicationIdx_X%(SR_img_dim - SR_filter_dim);
					}
				} else {
					P_applicationIdx_F += P_applicationIdx_Y/((SR_img_dim - SR_filter_dim)*(SR_img_dim - SR_filter_dim));
					P_applicationIdx_Y += P_applicationIdx_Y%((SR_img_dim - SR_filter_dim)*(SR_img_dim - SR_filter_dim));
				}
			}else{

					RecvReturnPause_applyfilter(CTL_RTN);
			}
		}
		break;
		case CTL_JOIN: {
			RecvReturnJoin_applyfilter();
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


