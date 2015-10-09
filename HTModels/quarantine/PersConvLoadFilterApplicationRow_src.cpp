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
CPersConvcachebottomapplication::PersConvcachebottomapplication()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case CONVLOADFILTER_ENTRY: {
			P_bottom_row_stride = SR_bottom_dim*2;
			P_bottom_channel_stride = P_bottom_row_stride*SR_bottom_dim;
			P_bottom_sample_stride = P_bottom_channel_stride*SR_bottom_channels;
			HtContinue(CONVLOADFILTER_COMPUTE_ADDR);
		}
		break;
		/*
		 * Insert padding 0's or not.
		 */
		case CONVLOADFILTER_COMPUTE_ADDR: {
			P_my_bottom_addr = SR_bottom_addr +
								PR_sample_idx*P_bottom_sample_stride +
								PR_i_c * PR_bottom_channel_stride +
								PR_y_s * PR_bottom_row_stride +
								PR_x_s*2;
			P_cache_idx = (PR_y_s+SR_p_y)*SR_d_f+(PR_x_s+SR_p_x);
			HtContinue(CONVLOADFILTER_COMPUTE_ALIGNED_ADDR);
		}
		break;
		/*
		 * Insert padding 0's or not.
		 */
		case CONVLOADFILTER_COMPUTE_ALIGNED_ADDR: {
			int offset = PR_my_bottom_addr%8;
			P_my_bottom_addr -= offset;
			P_bottom_read_idx = offset/2;
			HtContinue(CONVLOADFILTER_COMPUTE_ALIGNED_ADDR);
		}
		break;
		/*
		 * Determine what to do with next index.
		 */
		case CONVLOADFILTER_BRANCH: {
			if(PR_x_f >= SR_d_f){
				HtContinue(CONVLOADFILTER_RTN);
			}else{
				if(PR_x_s < 0 || PR_x_s >= SR_d_s){
					HtContinue(CONVLOADFILTER_PAD);
				}else{
					if(P_bottom_read_idx < 4){
						HtContinue(CONVLOADFILTER_STORE);
					}else{
						HtContinue(CONVLOADFILTER_READ);
					}
				}
			}
		}
		break;
		/*
		 * Insert a pad 0 and increment the index.
		 */
		case CONVLOADFILTER_PAD: {
			GW_bottom_diff_data[PR_cache_idx] = 0;
			PR_cache_idx ++;
			P_x_s++;
			P_x_f++;
			HtContinue(CONVLOADFILTER_BRANCH);
		}
		break;
		/*
		 * Read from memory
		 */
		case CONVLOADFILTER_READ: {
			BUSY_RETRY(ReadMemBusy());
			ReadMem_bottom_diff_data_raw(SR_my_bottom_addr + PR_bottom_raw_idx);
			ReadMemPause(CONVLOADFILTER_STORE);
		}
		break;
		/*
		 * Store a memory read element
		 */
		case CONVLOADFILTER_STORE: {
			GW_bottom_diff_data[PR_cache_idx] = (PR_bottom_diff_data_raw >> 16*PR_bottom_read_idx) & UINT16_MASK;
			P_bottom_read_idx++;
			P_x_s++;
			P_x_f++;
			P_bottom_raw_idx +=2;
			HtContinue(CONVLOADFILTER_BRANCH);
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


