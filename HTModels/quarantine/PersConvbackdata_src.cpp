/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersConvbackbias.h"

#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }
#define UINT16_MASK 0xFFFF


void
CPersConvbackbias::PersConvbackbias()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		//Setup
		case CONVBACKBIAS_ENTRY: {
			P_top_idx = 0;
			P_top_channel_idx = 0;
			P_bias_idx = 0;
			HtContinue(CONVBACKBIAS_INIT_BIAS_READ);
		}
		break;
		//Read bias diffs
		case CONVBACKBIAS_INIT_BIAS_READ: {
			BUSY_RETRY(ReadMemBusy());
			//printf("Bias read at %llx w/ offset %d\n",SR_bias_addr+PR_bias_idx*2,PR_bias_idx);
			ReadMem_bias_diff_raw(SR_bias_addr+PR_bias_idx*2);
			P_bias_cache_addr = SR_bias_addr+PR_bias_idx*2;
			P_bias_element_idx = 0;
			ReadMemPause(CONVBACKBIAS_INIT_BIAS_READ_STORE);
		}
		break;
		//Unpack bias diffs
		case CONVBACKBIAS_INIT_BIAS_READ_STORE: {
			//printf("Bias Read: %lx\n",PR_bias_diff_raw);
			P_bias_diff[3] = PR_bias_diff_raw & UINT16_MASK;
			P_bias_diff[2] = (PR_bias_diff_raw >> 16) & UINT16_MASK;
			P_bias_diff[1] = (PR_bias_diff_raw >> 32) & UINT16_MASK;
			P_bias_diff[0] = (PR_bias_diff_raw >> 48) & UINT16_MASK;
			HtContinue(CONVBACKBIAS_TOP_READ);
		}
		break;
		//Test top index for overflow => write and end algo
		case CONVBACKBIAS_TOP_TEST: {
			if(PR_top_idx < SR_size){
				HtContinue(CONVBACKBIAS_BIAS_TEST);
			}
			else{
				HtContinue(CONVBACKBIAS_BIAS_WRITE);
			}
		}
		break;
		//Read top data from memory.
		case CONVBACKBIAS_TOP_READ: {
			BUSY_RETRY(ReadMemBusy());
			//printf("write activation: %x <= %lx \n", PR_out_Addr.to_int(),PR_activation_group);
			ReadMem_top_data_raw(SR_top_addr+PR_top_idx*2);
			P_top_element_idx = 0;
			ReadMemPause(CONVBACKBIAS_TOP_READ_STORE);
		}
		break;
		//Unpack values from top read
		case CONVBACKBIAS_TOP_READ_STORE: {
			//printf("Top Read: %lx\n",PR_top_data_raw);
			P_top_data[3] = PR_top_data_raw & UINT16_MASK;
			P_top_data[2] = (PR_top_data_raw >> 16) & UINT16_MASK;
			P_top_data[1] = (PR_top_data_raw >> 32) & UINT16_MASK;
			P_top_data[0] = (PR_top_data_raw >> 48) & UINT16_MASK;
			HtContinue(CONVBACKBIAS_BIAS_TEST);
		}
		break;
		case CONVBACKBIAS_BIAS_TEST: {
			if(PR_bias_idx < SR_channels){
				HtContinue(CONVBACKBIAS_TOP_ELEMENT_TEST);
			}
			else{
				P_bias_idx = 0;
				HtContinue(CONVBACKBIAS_BIAS_WRITE);
			}
		}
		break;
		//Pack and write bias diffs
		case CONVBACKBIAS_BIAS_WRITE: {
			BUSY_RETRY(WriteMemBusy());
			//printf("write activation: %x <= %lx \n", PR_out_Addr.to_int(),PR_activation_group);
			uint64_t memWrData =  (uint64_t)PR_bias_diff[3];
			memWrData = (memWrData << 16) | (PR_bias_diff[2] &  UINT16_MASK);
			memWrData = (memWrData << 16) | (PR_bias_diff[1] &  UINT16_MASK);
			memWrData = (memWrData << 16) | (PR_bias_diff[0] &  UINT16_MASK);
			//printf("Bias Write: %lx\n",memWrData);
			WriteMem(PR_bias_cache_addr, memWrData);
			if(PR_top_idx < SR_size){
				WriteMemPause(CONVBACKBIAS_BIAS_READ);
			}
			else{
				WriteMemPause(CONVBACKBIAS_RTN);
			}
		}
		break;
		//Read bias diffs
		case CONVBACKBIAS_BIAS_READ: {
			BUSY_RETRY(ReadMemBusy());
			//printf("Bias read at %llx w/ offset %d\n",SR_bias_addr+PR_bias_idx*2,PR_bias_idx);
			ReadMem_bias_diff_raw(SR_bias_addr+PR_bias_idx*2);
			P_bias_cache_addr = SR_bias_addr+PR_bias_idx*2;
			P_bias_element_idx = 0;
			ReadMemPause(CONVBACKBIAS_BIAS_READ_STORE);
		}
		break;
		//Unpack bias diffs
		case CONVBACKBIAS_BIAS_READ_STORE: {
			//printf("Bias Read: %lx\n",PR_bias_diff_raw);
			P_bias_diff[3] = PR_bias_diff_raw & UINT16_MASK;
			P_bias_diff[2] = (PR_bias_diff_raw >> 16) & UINT16_MASK;
			P_bias_diff[1] = (PR_bias_diff_raw >> 32) & UINT16_MASK;
			P_bias_diff[0] = (PR_bias_diff_raw >> 48) & UINT16_MASK;
			HtContinue(CONVBACKBIAS_TOP_ELEMENT_TEST);
		}
		break;
		//Test if top index has gone past cached data
		case CONVBACKBIAS_TOP_ELEMENT_TEST: {
			if(PR_top_element_idx <4){
				HtContinue(CONVBACKBIAS_BIAS_ELEMENT_TEST);
			}
			else{
				HtContinue(CONVBACKBIAS_TOP_READ);
			}
		}
		break;
		//Test if bias index has gone past cached data
		case CONVBACKBIAS_BIAS_ELEMENT_TEST: {
			if(PR_bias_element_idx <4){
				HtContinue(CONVBACKBIAS_APPLY);
			}
			else{
				HtContinue(CONVBACKBIAS_BIAS_WRITE_PREP);
			}
		}
		break;
		//Compute the bias diff for current index
		case CONVBACKBIAS_APPLY: {
			//printf("Add %d to %d(%d) ,from %d(%d) %d\n",PR_top_data[PR_top_element_idx],PR_bias_idx,PR_bias_element_idx,PR_top_idx,PR_top_element_idx, P_top_channel_idx);
			P_bias_diff[PR_bias_element_idx] += PR_top_data[PR_top_element_idx];
			P_top_element_idx++;
			P_top_channel_idx++;
			P_top_idx++;
			HtContinue(CONVBACKBIAS_INCREMENT);
		}
		break;
		//Increment indexes.
		case CONVBACKBIAS_INCREMENT: {
			if(P_top_channel_idx >= SR_channel_stride){
				P_top_channel_idx=0;
				P_bias_idx++;
				P_bias_element_idx++;
			}
			HtContinue(CONVBACKBIAS_TOP_TEST);
		}
		break;
		//Complete
		case CONVBACKBIAS_RTN: {
			BUSY_RETRY(SendReturnBusy_conv_back_bias());
			SendReturn_conv_back_bias();
		}
		break;
		default:
			assert(0);
		}
	}
}
