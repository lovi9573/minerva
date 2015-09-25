/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersConvcachebottomapplication.h"


#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }
#define UINT16_MASK 0xFFFF


void
CPersConvcachebottomapplication::PersConvcachebottomapplication()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case CONVLOADFILTER_ENTRY: {
			printf("Filterload Entry\n");
			P_iter = 0;
			P_i = 0;
			P_j = 0;
			P_sample_idx = PR_rank;
			P_channel_idx = 0;
			P_bottom_raw_idx = 0;
			P_bottom_channel_stride = SR_bottom_dim*SR_bottom_dim;
			P_bottom_sample_stride = P_bottom_channel_stride*SR_bottom_channels;
			HtContinue(CONVLOADFILTER_READ);
		}
		break;
		case CONVLOADFILTER_READ: {
			BUSY_RETRY(ReadMemBusy());
			ReadMem_bottom_diff_data_raw(SR_bottom_addr +
									PR_sample_idx*PR_bottom_sample_stride +
									PR_channel_idx*PR_bottom_channel_stride +
									PR_filter_raw_idx);
			ReadMemPause(CONVLOADFILTER_STORE);
		}
		break;
		case CONVLOADFILTER_STORE: {
			GW_bottom_diff_data[PR_i][PR_j] = (PR_bottom_diff_data_raw >> 48) & UINT16_MASK;
			GW_bottom_diff_data[PR_i+(PR_j+1)/SR_filter_dim][(PR_j+1)%SR_filter_dim] = (PR_bottom_diff_data_raw >> 32) & UINT16_MASK;
			GW_bottom_diff_data[PR_i+(PR_j+2)/SR_filter_dim][(PR_j+2)%SR_filter_dim] = (PR_bottom_diff_data_raw >> 16) & UINT16_MASK;
			GW_bottom_diff_data[PR_i+(PR_j+3)/SR_filter_dim][(PR_j+3)%SR_filter_dim] = (PR_bottom_diff_data_raw >> 0) & UINT16_MASK;
			P_i += (PR_j+3)/SR_bottom_dim;
			P_j = (PR_j+3)%SR_bottom_dim;
			P_bottom_raw_idx +=8;
			HtContinue(CONVLOADFILTER_RAW_IDX_TEST);
		}
		break;
		case CONVLOADFILTER_RAW_IDX_TEST: {
			if(PR_bottom_raw_idx < P_bottom_channel_stride){
				HtContinue(CONVLOADFILTER_READ);
			}else{
				HtContinue(CONVLOADFILTER_DISPATCH);
			}
		}
		break;
		case CONVLOADFILTER_DISPATCH: {
			if(PR_i_bottom < SR_bottom_dim ){
				//printf("Compute        count:%d F:%d X:%d Y:%d I:%d %lu to %lu\n",PR_count.to_int(),PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I,P_Addresses[1],P_Addresses[0] );
				//printf("fork %lu; %lu\n",P_Addresses[1],P_Addresses[0]);
				BUSY_RETRY(SendCallBusy_conv_back_data());
				SendCallFork_conv_back_data(CONVLOADFILTER_JOIN, PR_i_bottom, PR_j_bottom );
				//rankStride leapfrog's me over other units to MY next application.
				P_i_bottom += (PR_j_bottom+3)/SR_bottom_dim;
				P_j_bottom = (PR_j_bottom+3)%SR_bottom_dim;
				HtContinue(CONVLOADFILTER_DISPATCH);
			}
			HtContinue(CONVLOADFILTER_JOIN);
		}
		break;
		case CONVLOADFILTER_JOIN: {
			RecvReturnJoin_conv_back_data();
			P_channel_idx++;
			P_filter_raw_idx = 0;
			HtContinue(CONV_CHANNEL_IDX_TEST);
		}
		break;
		case CONV_CHANNEL_IDX_TEST: {
			if(PR_channel_idx < SR_bottom_channels){
				HtContinue(CONVLOADFILTER_READ);
			}else{
				P_channel_idx = 0;
				P_sample_idx += PR_rankStride;
				HtContinue(CONV_SAMPLE_IDX_TEST);
			}
		}
		break;
		case CONV_SAMPLE_IDX_TEST: {
			if(P_sample_idx < SR_bottom_samples){
				HtContinue(CONVLOADFILTER_READ);
			}else{
				HtContinue(CONVLOADFILTER_RTN);
			}
		}
		break;
		case CONVLOADFILTER_RTN: {
			BUSY_RETRY(SendReturnBusy_conv_load_bottom_diff_layer());
			SendReturn_conv_load_bottom_diff_layer();
		}
		break;
		default:
			assert(0);
		}
	}
}


