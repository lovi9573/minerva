/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersConvloadfilter.h"


#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }
#define UINT16_MASK 0xFFFF


void
CPersConvloadfilter::PersConvloadfilter()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case CONVLOADFILTER_ENTRY: {
			printf("Filterload Entry\n");
			P_iter = 0;
			P_read_idx = 0;
			P_filter_raw_idx = 0;
			P_filter_stride = SR_filter_dim*SR_filter_dim*SR_bottom_channels;
			HtContinue(CONVLOADFILTER_READ);
		}
		break;
		/*
		 * Read a 64-bit chunk of memory
		 */
		case CONVLOADFILTER_READ: {
			BUSY_RETRY(ReadMemBusy());
			ReadMem_filter_data_raw(SR_filter_addr + PR_iter*PR_filter_stride + PR_filter_raw_idx);
			ReadMemPause(CONVLOADFILTER_STORE);
		}
		break;
		/*
		 * Store a 16-bit element into cache
		 */
		case CONVLOADFILTER_STORE: {
			GW_filter_data[PR_iter][PR_read_idx] = (PR_filter_data_raw >> 16*PR_read_idx) & UINT16_MASK;
			P_read_idx ++;
			P_filter_raw_idx +=2;
			HtContinue(CONVLOADFILTER_FILTER_RAW_IDX_TEST);
		}
		break;
		/*
		 * Check if another 16 bit element can be cached.
		 */
		case CONVLOADFILTER_READ_IDX_TEST: {
			if(PR_filter_raw_idx < PR_filter_stride && PR_read_idx < 4){
				HtContinue(CONVLOADFILTER_STORE);
			}else{
				P_filter_raw_idx = 0;
				HtContinue(CONVLOADFILTER_DISPATCH);
			}
		}
		break;
		/*
		 * Check if another 64-bit chunck can be read.
		 */
		case CONVLOADFILTER_FILTER_RAW_IDX_TEST: {
			if(PR_filter_raw_idx < PR_filter_stride){
				HtContinue(CONVLOADFILTER_READ);
			}else{
				P_filter_raw_idx = 0;
				P_iter++;
				HtContinue(CONVLOADFILTER_DISPATCH);
			}
		}
		break;
		/*
		 * Filter cache is completed, dispatch the job
		 */
		case CONVLOADFILTER_DISPATCH: {
			if(PR_task == CONV_BACKWARD_DATA){
				//printf("Compute        count:%d F:%d X:%d Y:%d I:%d %lu to %lu\n",PR_count.to_int(),PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I,P_Addresses[1],P_Addresses[0] );
				//printf("fork %lu; %lu\n",P_Addresses[1],P_Addresses[0]);
				//BUSY_RETRY(SendCallBusy_conv_back_data());
				//SendCallFork_conv_back_data(CONVLOADFILTER_JOIN, SR_outAddr + PR_outAddrOffset );
				HtContinue(CONVLOADFILTER_JOIN);

			}else{
				//BUSY_RETRY(SendCallBusy_conv_back_filter());
				//SendCall_conv_load_top_diff_layer(PR_task);
				//printf("increment F:%d X:%d Y:%d I:%d\n",PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I);
				HtContinue(CONVLOADFILTER_ITER_TEST);
			}
		}
		break;
		/*
		 * Join the dispatch
		 */
		case CONVLOADFILTER_JOIN: {
			//RecvReturnJoin_();
		}
		break;
		/*
		 * Test if another filter can be loaded.
		 */
		case CONVLOADFILTER_ITER_TEST: {
			if(PR_iter < PR_num_filters_to_iterate){
				HtContinue(CONVLOADFILTER_READ);
			}else{
				HtContinue(CONVLOADFILTER_RTN);
			}
		}
		break;
		case CONVLOADFILTER_RTN: {
			BUSY_RETRY(SendReturnBusy_conv_load_filter());
			SendReturn_conv_load_filter();
		}
		break;
		default:
			assert(0);
		}
	}
}


