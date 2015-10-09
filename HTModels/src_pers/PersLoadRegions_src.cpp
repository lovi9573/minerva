/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersLoadregions.h"


#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }
#define UINT16_MASK 0xFFFF

/*
 * x_s: x coordinate in the sample
 * y_s: y coordinate in the sample
 * x_f: x coordinate in the filter
 * y_f: y coordinate in the filter
 * i_r: index into the read data(uint16_t[4] 64-bit packed)
 * i_m: index into the memory (read/write mem)
 * i_c: index into the cache (global var)
 */

void
CPersLoadregions::PersLoadregions()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case LOADREGIONS_ENTRY: {
			if(PR_in_sample){
				P_sample_idx = PR_rank;
			}else{
				P_sample_idx = 0;
			}
			HtContinue(LOADREGIONS_INIT);
		}
		break;
		/*
		 * Initialize for new sample
		 */
		case LOADREGIONS_INIT: {
			P_x_s = -SR_p_x;
			P_y_s = -SR_p_y;
			P_y_f = 0;
			P_i_c = 0;
			if(PR_sample_idx < SR_bottom_samples){
				HtContinue(LOADREGIONS_LOAD_ROW);
			}else{
				HtContinue(LOADREGIONS_RTN);
			}
		}
		break;
		/*
		 * Fork a load for each row.
		 */
		case LOADREGIONS_LOAD_ROW: {
			//BUSY_RETRY(SendCallBusy_conv_load_filter_application_row());
			printf("\tload row at (%d,%d,%d,%d) (x,y,c,s)\n", PR_x_s, PR_y_s+PR_y_f, PR_i_c,PR_sample_idx);
			BUSY_RETRY(SendCallBusy_load_filter_application_row());
			SendCallFork_load_filter_application_row(LOADREGIONS_JOIN, P_sample_idx,PR_i_c , PR_y_s+PR_y_f, PR_x_s,PR_y_f );
			P_y_f++;
			HtContinue(LOADREGIONS_TEST_Y_F);
		}
		break;
		/*
		 * Test filter y dimension for fork of next row
		 */
		case LOADREGIONS_TEST_Y_F: {
			if(PR_y_f < SR_d_f){
				HtContinue(LOADREGIONS_LOAD_ROW);
			}else{
				P_i_c++;
				P_y_f -= SR_d_f;
				HtContinue(LOADREGIONS_TEST_I_C);
			}
		}
		break;
		/*
		 * Test filter y dimension for fork of next row
		 */
		case LOADREGIONS_TEST_I_C: {
			if(PR_i_c < SR_d_c){
				HtContinue(LOADREGIONS_LOAD_ROW);
			}else{
				HtContinue(LOADREGIONS_WAIT);
			}
		}
		break;
		/*
		 * Join the row loads
		 */
		case LOADREGIONS_JOIN: {
			RecvReturnJoin_load_filter_application_row();
			//HtContinue(LOADREGIONS_CALL_KERNEL);
		}
		break;
		/*
		 * Filter application cached, Call the kernel
		 */
		case LOADREGIONS_WAIT: {
			RecvReturnPause_load_filter_application_row(LOADREGIONS_CALL_KERNEL);
		}
		break;
		/*
		 * Filter application cached, Call the kernel
		 */
		case LOADREGIONS_CALL_KERNEL: {
			BUSY_RETRY(SendCallBusy_load_filters());
			printf("Call load filters w/ task %d\n",PR_task);
			SendCall_load_filters(LOADREGIONS_TEST_X_S, PR_task );
			P_x_s += SR_s_x;
			P_y_f = 0;
			P_i_c = 0;
			//HtContinue(LOADREGIONS_TEST_X_S);
		}
		//TODO: write back the cache after kernel call.
		break;
		/*
		 * Test if next application will fit in x dimension of sample
		 */
		case LOADREGIONS_TEST_X_S: {
			if(PR_x_s + SR_d_f > SR_d_s + SR_p_x){
				P_x_s = -SR_p_x;
				P_y_s += SR_s_y;
			}
			HtContinue(LOADREGIONS_TEST_Y_S);
		}
		break;
		/*
		 * Test if next application will fit in y dimension of sample
		 */
		case LOADREGIONS_TEST_Y_S: {
			if(PR_y_s + SR_d_f < SR_d_s + SR_p_y){
				HtContinue(LOADREGIONS_LOAD_ROW);
			}else{
				HtContinue(LOADREGIONS_NEXT_SAMPLE);
			}
		}
		break;
		/*
		 * Move to next sample.
		 */
		case LOADREGIONS_NEXT_SAMPLE: {
			if(! PR_in_sample ){
				P_sample_idx += PR_rankStride;
				HtContinue(LOADREGIONS_INIT);
			}else{
				HtContinue(LOADREGIONS_RTN);
			}
		}
		break;
		/*
		 * Return.
		 */
		case LOADREGIONS_RTN: {
			BUSY_RETRY(SendReturnBusy_conv_load_filter_applications());
			SendReturn_conv_load_filter_applications();
		}
		break;
		default:
			assert(0);
		}
	}
}


