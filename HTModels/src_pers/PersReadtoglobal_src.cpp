/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersReadtoglobal.h"


#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }
#define UINT16_MASK 0xFFFF

/*********************************
 * read n bytes into global cache
 * global cache to write to is selectable with which_mem
 *********************************/
void CPersReadtoglobal::PersReadtoglobal()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case READTOGLOBAL_ENTRY: {
			printf("\t\t\tReadToGlobal Entry\n");
			int byte_misalignment = PR_data_addr%8;
			P_data_raw_idx = byte_misalignment/2;
			P_data_addr -= byte_misalignment;
			P_data_addr_offset = 0;
			HtContinue(READTOGLOBAL_READ);
		}
		break;
		/*
		 * Read a 64-bit chunk of memory
		 */
		case READTOGLOBAL_READ: {
			BUSY_RETRY(ReadMemBusy());
			printf("\t\t\tRead From %lu\n",(PR_data_addr + PR_data_addr_offset));
			ReadMem_data_raw(PR_data_addr + PR_data_addr_offset);
			if(PR_which_mem == GLOBAL_SAMPLE_MEM){
				P_gb_addr1 = PR_global_idx;
				P_gb_addr2 = PR_i_c;
			}
			else if(PR_which_mem == GLOBAL_FILTER_MEM){
				P_gf_addr1 = PR_global_idx;
				P_gf_addr2 = PR_i_c;
			}
			ReadMemPause(READTOGLOBAL_STORE);
		}
		break;
		/*
		 * Store a 16-bit element into cache
		 */
		case READTOGLOBAL_STORE: {
			printf("\t\t\tStore to global %d idx: %d\n",PR_which_mem, PR_global_idx);
			if(PR_which_mem == GLOBAL_SAMPLE_MEM){
				GW_bottom_data.write_addr(PR_global_idx,PR_i_c);
				GW_bottom_data = (PR_data_raw >> 16*PR_data_raw_idx) & UINT16_MASK;
			}
			else if(PR_which_mem == GLOBAL_FILTER_MEM){
				GW_filter_data.write_addr(PR_global_idx,PR_i_c);
				GW_filter_data = (PR_data_raw >> 16*PR_data_raw_idx) & UINT16_MASK;
			}
			P_data_raw_idx ++;
			P_global_idx++;
			HtContinue(READTOGLOBAL_READ_IDX_TEST);
		}
		break;
		/*
		 * Check if another 16 bit element can be cached.
		 */
		case READTOGLOBAL_READ_IDX_TEST: {
			if(PR_global_idx < PR_elements && P_data_raw_idx < 4 ){
				HtContinue(READTOGLOBAL_STORE);
			}else{
				P_data_addr_offset +=8;
				P_data_raw_idx = 0;
				HtContinue(READTOGLOBAL_FILTER_RAW_IDX_TEST);
			}
		}
		break;
		/*
		 * Check if another 64-bit chunck can be read.
		 */
		case READTOGLOBAL_FILTER_RAW_IDX_TEST: {
			if(PR_global_idx < PR_elements){
				HtContinue(READTOGLOBAL_READ);
			}else{
				HtContinue(READTOGLOBAL_RTN);
			}
		}
		break;
		case READTOGLOBAL_RTN: {
			BUSY_RETRY(SendReturnBusy_read_to_global());
			SendReturn_read_to_global();
		}
		break;
		default:
			assert(0);
		}
	}
}


