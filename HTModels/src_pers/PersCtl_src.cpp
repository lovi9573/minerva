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
			P_count = 0;
			P_outAddrOffset = 0;
			//Determine MY coordinate in X,Y,Filter space based on MY unit rank.
			P_applicationIdx_F=P_rank*4%SR_filter_num;  //Each takes 4 filters
			P_applicationIdx_X=((P_rank*4/SR_filter_num)*SR_stride)%SR_img_dim;
			P_applicationIdx_Y=(((P_rank*4/SR_filter_num)*SR_stride)/SR_img_dim)*SR_stride%SR_img_dim;
			P_applicationIdx_I=(((P_rank*4/SR_filter_num)*SR_stride)/SR_img_dim)*SR_stride/SR_img_dim;
			HtContinue(CTL_CHECK_I);
		}
		break;
		case CTL_CHECK_I: {
			if(PR_applicationIdx_I  < SR_img_num){
				HtContinue(CTL_CHECK_Y);
			}else{
				RecvReturnPause_scattergatherfiltergroup(CTL_RTN);
				//HtContinue(CTL_RTN);
			}
		}
		break;
		case CTL_CHECK_Y: {
			if(P_applicationIdx_Y  < SR_img_dim - SR_filter_dim + SR_stride){
				printf("Y: %d\n",PR_applicationIdx_Y);
				HtContinue(CTL_CHECK_X);
			}else{//overflow Y index
				P_applicationIdx_I = PR_applicationIdx_I + PR_applicationIdx_Y/(SR_img_dim - SR_filter_dim);
				P_applicationIdx_Y = PR_applicationIdx_Y%(SR_img_dim - SR_filter_dim + SR_stride);
				HtContinue(CTL_CHECK_I);
			}
		}
		break;
		case CTL_CHECK_X: {
			if (PR_applicationIdx_X <  SR_img_dim - SR_filter_dim + SR_stride) {
				printf("\tX: %d\n",PR_applicationIdx_X);
				HtContinue(CTL_CHECK_F);
			} else {//overflow X index
				P_applicationIdx_Y = PR_applicationIdx_Y + PR_applicationIdx_X/(SR_img_dim - SR_filter_dim)*SR_stride;
				P_applicationIdx_X = PR_applicationIdx_X%(SR_img_dim - SR_filter_dim + SR_stride);
				HtContinue(CTL_CHECK_Y);
			}
		}
		break;
		case CTL_CHECK_F: {
			if(PR_applicationIdx_F  < SR_filter_num){
				HtContinue(CTL_COLLECT);
			}else{ //overflow filter index
				P_applicationIdx_X = PR_applicationIdx_X + PR_applicationIdx_F/SR_filter_num*SR_stride;
				P_applicationIdx_F = P_applicationIdx_F%SR_filter_num;
				HtContinue(CTL_CHECK_X);
			}
		}
		break;
		case CTL_COLLECT: {
			//printf("adress collect count:%d F:%d X:%d Y:%d I:%d\n",PR_count.to_int(),PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I );
			//Valid application of a filter
			P_Addresses[PR_count*2] = SR_imgAddr + 2*(
													P_applicationIdx_I*SR_img_dim*SR_img_dim*SR_img_channels +
													P_applicationIdx_Y*SR_img_dim*SR_img_channels +
													P_applicationIdx_X*SR_img_channels
													); //Image patch base address
			P_Addresses[PR_count*2+1] = SR_filterAddr + 2*(PR_applicationIdx_F *SR_filter_dim*SR_filter_dim*SR_img_channels ); //Filter base address
			P_count = PR_count +1;
			HtContinue(CTL_COMPUTE);
		}
		break;
		case CTL_COMPUTE: {
			if(PR_count >= 4){
				//printf("Compute        count:%d F:%d X:%d Y:%d I:%d %lu to %lu\n",PR_count.to_int(),PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I,P_Addresses[1],P_Addresses[0] );
				//printf("fork %lu; %lu\n",P_Addresses[1],P_Addresses[0]);
				BUSY_RETRY(SendCallBusy_scattergatherfiltergroup());
				SendCallFork_scattergatherfiltergroup(CTL_JOIN, P_Addresses[0],P_Addresses[1],
																P_Addresses[2],P_Addresses[3],
																P_Addresses[4],P_Addresses[5],
																P_Addresses[6],P_Addresses[7],
																SR_outAddr + PR_outAddrOffset );

				P_count=0;
				P_outAddrOffset = PR_outAddrOffset + 8;
				//rankStride leapfrog's me over other units to MY next application.
				P_applicationIdx_F =  PR_applicationIdx_F + (P_rankStride-1)*4+1;
			}else{
				P_applicationIdx_F = PR_applicationIdx_F + 1;
				//printf("increment F:%d X:%d Y:%d I:%d\n",PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I);
			}
			HtContinue(CTL_CHECK_F);
		}
		break;
		case CTL_JOIN: {
			RecvReturnJoin_scattergatherfiltergroup();
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


