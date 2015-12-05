/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersLoadfilters.h"


#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }
#define UINT16_MASK 0xFFFF

/*********************************
 * Iterate through and Load Filters into global cache
 * For each filter loaded call a kernel to apply that filter to data.
 * Serially call back data filters to avoid write conflicts
 * Fork call back filter kernels
 *********************************/
void CPersLoadfilters::PersLoadfilters()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		/*
		 * Initialize
		 */
		case LOADFILTERS_ENTRY: {
			P_i_f = 0;
			P_global_idx =0;
			P_filter_raw_idx = 0;
			P_filter_addr_offset = 0;
			P_DS_f = SR_filter_dim*SR_filter_dim*SR_bottom_channels;
			HtContinue(LOADFILTERS_READ);
		}
		break;
		/*
		 * Load filter(s) into global cache
		 */
		case LOADFILTERS_READ: {
			BUSY_RETRY(SendCallBusy_read_to_global());
			printf("\tReading filter to global mem\n");
			SendCall_read_to_global(LOADFILTERS_DISPATCH,
									GLOBAL_FILTER_MEM,
									SR_filter_addr + PR_i_f*PR_DS_f + PR_filter_addr_offset,
									PR_DS_f,
									0,
									PR_i_f);
			P_i_f++;
			//ReadMemPause(LOADFILTERS_STORE);
		}
		break;
		/*
		 * Filter cache is completed, dispatch the job
		 */
		case LOADFILTERS_DISPATCH: {
//			if(PR_task == CONV_BACKWARD_DATA){
//				printf("\tcall module to update data using filter %d\n",PR_i_f);
//				BUSY_RETRY(SendCallBusy_conv_back_data());
//				SendCall_conv_back_data();
//				//printf("increment F:%d X:%d Y:%d I:%d\n",PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I);
//				HtContinue(LOADFILTERS_ITER_TEST);
//			}else{
//				//printf("Compute        count:%d F:%d X:%d Y:%d I:%d %lu to %lu\n",PR_count.to_int(),PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I,P_Addresses[1],P_Addresses[0] );
//				printf("\tfork module to update filter %d\n",PR_i_f);
//				BUSY_RETRY(SendCallBusy_conv_back_filter());
//				SendCallFork_conv_back_filter(LOADFILTERS_JOIN, PR_i_f );
//				HtContinue(LOADFILTERS_ITER_TEST);
//			}
			BUSY_RETRY(SendCallBusy_conv_load_filter_applications());
			SendCall_conv_load_filter_applications(LOADFILTERS_ITER_TEST,PR_rank, PR_rankStride, false, PR_task);
			//printf("increment F:%d X:%d Y:%d I:%d\n",PR_applicationIdx_F,PR_applicationIdx_X,PR_applicationIdx_Y,PR_applicationIdx_I);
			HtContinue(LOADFILTERS_ITER_TEST);
		}
		break;
		/*
		 * Join the dispatch
		 */
		case LOADFILTERS_JOIN: {
			//RecvReturnJoin_conv_back_filter();
		}
		break;
		/*
		 * Test if another filter can be loaded.
		 */
		case LOADFILTERS_ITER_TEST: {
			printf("load filter number %d/%d\n",PR_i_f,SR_num_filters);
			if(PR_i_f < SR_num_filters){
				HtContinue(LOADFILTERS_READ);
			}else{
				HtContinue(LOADFILTERS_RTN);
			}
		}
		break;
		case LOADFILTERS_RTN: {
			BUSY_RETRY(SendReturnBusy_load_filters());
			SendReturn_load_filters();
		}
		break;
		default:
			assert(0);
		}
	}
}


