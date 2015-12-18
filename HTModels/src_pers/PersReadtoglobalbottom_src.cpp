/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersReadtoglobalbottom.h"


#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }
#define UINT16_MASK 0xFFFF

/*********************************
 * read n bytes into global cache
 * global cache to write to is selectable with which_mem
 *********************************/
void CPersReadtoglobalbottom::PersReadtoglobalbottom()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case READTOGLOBALBOTTOM_ENTRY: {
			printf("\t\t\tReadToGlobalBottom Entry\n");
			int byte_misalignment = PR_data_addr%8;
			P_data_raw_idx = byte_misalignment/2;
			P_data_addr -= byte_misalignment;
			P_data_addr_offset = 0;
			HtContinue(READTOGLOBALBOTTOM_READ);
		}
		break;
		/*
		 * Read a 64-bit chunk of memory
		 */
		case READTOGLOBALBOTTOM_READ: {
			BUSY_RETRY(ReadMemBusy());
			printf("\t\t\tRead From %lu\n",(PR_data_addr + PR_data_addr_offset));
			ReadMem_data_raw(PR_data_addr + PR_data_addr_offset);
			P_gb_addr1 = PR_global_idx;
			P_gb_addr2 = PR_i_c;
			ReadMemPause(READTOGLOBALBOTTOM_STORE);
		}
		break;
		/*
		 * Store a 16-bit element into cache
		 */
		case READTOGLOBALBOTTOM_STORE: {
			printf("\t\t\tStore to global bottom idx: %d\n", PR_global_idx);
			GW_bottom_data.write_addr(PR_global_idx,PR_i_c);
			GW_bottom_data = (PR_data_raw >> 16*PR_data_raw_idx) & UINT16_MASK;
			P_data_raw_idx ++;
			P_global_idx++;
			HtContinue(READTOGLOBALBOTTOM_READ_IDX_TEST);
		}
		break;
		/*
		 * Check if another 16 bit element can be cached.
		 */
		case READTOGLOBALBOTTOM_READ_IDX_TEST: {
			if(PR_global_idx < PR_elements && P_data_raw_idx < 4 ){
				HtContinue(READTOGLOBALBOTTOM_STORE);
			}else{
				P_data_addr_offset +=8;
				P_data_raw_idx = 0;
				HtContinue(READTOGLOBALBOTTOM_FILTER_RAW_IDX_TEST);
			}
		}
		break;
		/*
		 * Check if another 64-bit chunck can be read.
		 */
		case READTOGLOBALBOTTOM_FILTER_RAW_IDX_TEST: {
			if(PR_global_idx < PR_elements){
				HtContinue(READTOGLOBALBOTTOM_READ);
			}else{
				HtContinue(READTOGLOBALBOTTOM_RTN);
			}
		}
		break;
		case READTOGLOBALBOTTOM_RTN: {
			BUSY_RETRY(SendReturnBusy_read_to_global_bottom());
			SendReturn_read_to_global_bottom();
		}
		break;
		default:
			assert(0);
		}
	}
}


