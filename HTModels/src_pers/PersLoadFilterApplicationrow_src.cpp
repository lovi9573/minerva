/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersLoadfilterapplicationrow.h"


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
CPersLoadfilterapplicationrow::PersLoadfilterapplicationrow()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		/*
		 * Initialize for new sample
		 */
		case CONVLOADFILTERAPPLICATIONROW_ENTRY: {
			P_x_f = 0;
			HtContinue(CONVLOADFILTERAPPLICATIONROW_TEST_X_S);
		}
		break;
		/*
		 * Test x bounds
		 */
		case CONVLOADFILTERAPPLICATIONROW_TEST_X_S: {
			printf("\t\tload row x_f(d_f): %d(%d), x_s: %d(%d)\n",PR_x_f, SR_d_f, PR_x_s, SR_d_s );
			if(PR_x_f < SR_d_f){
				if(PR_x_s < 0 || PR_x_s >= SR_d_s || PR_y_s < 0 || PR_y_s > SR_d_s){
					P_g_addr1 = PR_x_f+PR_y_f*SR_d_f;
					P_g_addr2 = PR_c_s;
					HtContinue(CONVLOADFILTERAPPLICATIONROW_PAD);
				}else{
					HtContinue(CONVLOADFILTERAPPLICATIONROW_COMPUTE_ADDR);
				}
			}else{
				HtContinue(CONVLOADFILTERAPPLICATIONROW_RTN);
			}
		}
		break;
		/*
		 * Fork a load for each row.
		 */
		case CONVLOADFILTERAPPLICATIONROW_PAD :{
			//BUSY_RETRY(SendCallBusy_conv_load_filter_application_row());
			//SendCallFork_conv_load_filter_application_row(CONVLOADFILTERAPPLICATIONSAPPLICATIONS_JOIN, P_sample_idx, PR_x_s, PR_y_s+PR_y_f, PR_y_f );
			GW_bottom_data.write_addr(PR_g_addr1, PR_g_addr2);
			GW_bottom_data = 0;
			P_x_s++;
			P_x_f++;
			HtContinue(CONVLOADFILTERAPPLICATIONROW_TEST_X_S);
		}
		break;
		case CONVLOADFILTERAPPLICATIONROW_COMPUTE_ADDR: {
			//Compute start address.
			P_bottom_addr_offset = PR_i_s*SR_DS_s+
								   PR_c_s*SR_DS_c+
								   PR_y_s*SR_DS_y+
								   PR_x_s;
			if(SR_d_f-PR_x_f < SR_d_s - PR_x_s){
				P_read_size = SR_d_f-PR_x_f;
			}else{
				P_read_size = SR_d_s - PR_x_s;
			}
			HtContinue(CONVLOADFILTERAPPLICATIONROW_READ);
		}
		break;
		/*
		 * Test filter y dimension for fork of next row
		 */
		case CONVLOADFILTERAPPLICATIONROW_READ: {
			printf("\t\tRead %d elements from %d\n", PR_read_size, PR_bottom_addr_offset.to_int());
			BUSY_RETRY(SendCallBusy_read_to_global_bottom());
			SendCall_read_to_global_bottom(CONVLOADFILTERAPPLICATIONROW_TEST_X_S,
									SR_bottom_addr+PR_bottom_addr_offset,
									PR_read_size,
									PR_x_s + PR_x_f + PR_y_s*SR_d_s + PR_y_f*SR_d_f,
									PR_c_s);
			P_x_f += P_read_size;
			P_x_s += P_read_size;
			//HtContinue(CONVLOADFILTERAPPLICATIONROW_TEST_X_S);
		}
		break;
		/*
		 * Return.
		 */
		case CONVLOADFILTERAPPLICATIONROW_RTN: {
			BUSY_RETRY(SendReturnBusy_load_filter_application_row());
			SendReturn_load_filter_application_row();
		}
		break;
		default:
			assert(0);
		}
	}
}


