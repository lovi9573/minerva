/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersConvfwd.h"


#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }


void
CPersConvfwd::PersConvfwd()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case CONVFWD_ENTRY: {
			printf("Control Entry\n");
			P_count = 0;
			P_outAddrOffset = 0;
			S_dCoordinates = SR_img_dim - SR_filter_dim + SR_stride;
			S_dimY = (uint32_t)(SR_img_dim*SR_img_dim*SR_img_channels);
			S_dimX = (uint32_t)(SR_img_dim*SR_img_channels);
			S_dimF = (uint32_t)(SR_filter_dim*SR_filter_dim*SR_img_channels);
			//Determine MY coordinate in X,Y,Filter space based on MY unit rank.
			P_applicationIdx_F=(uint32_t)(P_rank*4%SR_filter_num);  //Each takes 4 filters
			P_applicationIdx_X=((P_rank*4/SR_filter_num)*SR_stride)%SR_img_dim;
			P_applicationIdx_Y=(((P_rank*4/SR_filter_num)*SR_stride)/SR_img_dim)*SR_stride%SR_img_dim;
			P_applicationIdx_I=(uint32_t)((((P_rank*4/SR_filter_num)*SR_stride)/SR_img_dim)*SR_stride/SR_img_dim);
			HtContinue(CONVFWD_CHECK_I);
		}
		break;
		case CONVFWD_CHECK_I: {
			if(PR_applicationIdx_I  < SR_img_num){
				HtContinue(CONVFWD_CHECK_Y);
			}else{
				RecvReturnPause_cluster(CONVFWD_RTN);
				//HtContinue(CONVFWD_RTN);
			}
		}
		break;
		case CONVFWD_CHECK_Y: {
			if(P_applicationIdx_Y  < SR_dCoordinates){
				//printf("Y: %d\n",PR_applicationIdx_Y);
				HtContinue(CONVFWD_CHECK_X);
			}else{//overflow Y index
				P_applicationIdx_I = PR_applicationIdx_I + 1;
				P_applicationIdx_Y = PR_applicationIdx_Y - (SR_dCoordinates);
				HtContinue(CONVFWD_CHECK_I);
			}
		}
		break;
		case CONVFWD_CHECK_X: {
			if (PR_applicationIdx_X <  SR_dCoordinates) {
				//printf("\tX: %d\n",PR_applicationIdx_X);
				HtContinue(CONVFWD_CHECK_F);
			} else {//overflow X index
				P_applicationIdx_Y = PR_applicationIdx_Y + SR_stride;
				P_applicationIdx_X = PR_applicationIdx_X -(SR_dCoordinates);
				HtContinue(CONVFWD_CHECK_Y);
			}
		}
		break;
		case CONVFWD_CHECK_F: {
			if(PR_applicationIdx_F  < SR_filter_num){
				HtContinue(CONVFWD_COLLECT);
			}else{ //overflow filter index
				P_applicationIdx_X = PR_applicationIdx_X + SR_stride;
				P_applicationIdx_F = P_applicationIdx_F - SR_filter_num;
				HtContinue(CONVFWD_CHECK_X);
			}
		}
		break;
		case CONVFWD_COLLECT: {
			//printf("adress collect count:%d F:%d X:%d Y:%d I:%d\n",PR_count.to_int(),PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I );
			//Valid application of a filter
			P_Addresses[PR_count*2] = SR_imgAddr + (MemAddr_t)(2*(
													P_applicationIdx_I*SR_dimY +
													P_applicationIdx_Y*SR_dimX +
													P_applicationIdx_X*SR_img_channels)
													); //Image patch base address
			P_Addresses[PR_count*2+1] = SR_filterAddr + (MemAddr_t)(2*(PR_applicationIdx_F *SR_dimF )); //Filter base address
			HtContinue(CONVFWD_COMPUTE);
		}
		break;
		case CONVFWD_COMPUTE: {
			if(PR_count == 3){
				//printf("Compute        count:%d F:%d X:%d Y:%d I:%d %lu to %lu\n",PR_count.to_int(),PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I,P_Addresses[1],P_Addresses[0] );
				//printf("fork %lu; %lu\n",P_Addresses[1],P_Addresses[0]);
				BUSY_RETRY(SendCallBusy_cluster());
				SendCallFork_cluster(CONVFWD_JOIN, P_Addresses[0],P_Addresses[1],
																P_Addresses[2],P_Addresses[3],
																P_Addresses[4],P_Addresses[5],
																P_Addresses[6],P_Addresses[7],
																SR_outAddr + PR_outAddrOffset );

				P_count=0;
				P_outAddrOffset = PR_outAddrOffset + 8;
				//rankStride leapfrog's me over other units to MY next application.
				P_applicationIdx_F =  PR_applicationIdx_F + (PR_rankStride-1)*4+1;
			}else{
				P_count = PR_count +1;
				P_applicationIdx_F = PR_applicationIdx_F + 1;
				//printf("increment F:%d X:%d Y:%d I:%d\n",PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I);
			}
			HtContinue(CONVFWD_CHECK_F);
		}
		break;
		case CONVFWD_JOIN: {
			RecvReturnJoin_cluster();
		}
		break;

		case CONVFWD_RTN: {
			BUSY_RETRY(SendReturnBusy_conv_fwd());
			SendReturn_conv_fwd();
		}
		break;
		default:
			assert(0);
		}
	}
}


