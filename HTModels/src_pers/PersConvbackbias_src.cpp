/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersConvbackbias.h"

#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }



void
CPersConvbackbias::PersConvbackbias()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case CTL_ENTRY: {
			P_top_idx = 0;
			P_top_channel_idx = 0;
			P_bias_idx = PR_channels +1;
			HtContinue(CTL_TOP_READ);
		}
		break;
		case CTL_TOP_TEST: {
			if(PR_top_idx < PR_size){
				HtContinue(CTL_BIAS_TEST);
			}
			else{
				HtContine(CTL_BIAS_WRITE);
			}
		}
		break;
		case CTL_TOP_READ: {
			BUSY_RETRY(ReadMemBusy());
			//printf("write activation: %x <= %lx \n", PR_out_Addr.to_int(),PR_activation_group);
			ReadMem_top_data(PR_top_addr+PR_top_idx/4);
			P_top_element_idx = 0;
			ReadMemPause(CTL_TOP_READ_STORE);
		}
		break;
		case CTL_TOP_READ_STORE: {
			P_top_data[0] = PR_top_data_raw.i0;
			P_top_data[1] = PR_top_data_raw.i1;
			P_top_data[2] = PR_top_data_raw.i2;
			P_top_data[3] = PR_top_data_raw.i3;
			ReadMemPause(CTL_BIAS_TEST);
		}
		break;
		case CTL_BIAS_TEST: {
			if(PR_bias_idx < PR_channels){
				HtContinue(CTL_TOP_ELEMENT_TEST);
			}
			else{
				P_bias_idx = 0;
				HtContinue(CTL_BIAS_WRITE_PREP);
			}
		}
		break;
		case CTL_BIAS_WRITE_PREP: {
			P_bias_data_raw.i0 = P_bias_data[0];
			P_bias_data_raw.i1 = P_bias_data[0];
			P_bias_data_raw.i2 = P_bias_data[0];
			P_bias_data_raw.i3 = P_bias_data[0];
			WriteMemPause(CTL_BIAS_WRITE);
		}
		break;
		case CTL_BIAS_WRITE: {
			BUSY_RETRY(WriteMemBusy());
			//printf("write activation: %x <= %lx \n", PR_out_Addr.to_int(),PR_activation_group);
			WriteMem(PR_bias_cache_addr, PR_bias_data);
			if(PR_top_idx < PR_size){
				WriteMemPause(CTL_BIAS_READ);
			}
			else{
				WriteMemPause(CTL_RTN);
			}
		}
		break;
		case CTL_BIAS_READ: {
			BUSY_RETRY(ReadMemBusy());
			//printf("write activation: %x <= %lx \n", PR_out_Addr.to_int(),PR_activation_group);
			ReadMem_bias_data(PR_bias_addr+PR_bias_idx/4);
			P_bias_cache_addr = PR_bias_addr+PR_bias_idx/4;
			P_bias_element_idx = 0;
			ReadMemPause(CTL_BIAS_READ_STORE);
		}
		break;
		case CTL_BIAS_READ_STORE: {
			P_bias_data[0] = PR_bias_data_raw.i0;
			P_bias_data[1] = PR_bias_data_raw.i1;
			P_bias_data[2] = PR_bias_data_raw.i2;
			P_bias_data[3] = PR_bias_data_raw.i3;
			ReadMemPause(CTL_TOP_ELEMENT_TEST);
		}
		break;
		case CTL_TOP_ELEMENT_TEST: {
			if(PR_element_idx <4){
				HtContinue(CTL_BIAS_ELEMENT_TEST);
			}
			else{
				HtContinue(CTL_TOP_READ);
			}
		}
		break;
		case CTL_BIAS_ELEMENT_TEST: {
			if(PR_bias_element_idx <4){
				HtContinue(CTL_APPLY);
			}
			else{
				HtContinue(CTL_BIAS_WRITE_PREP);
			}
		}
		break;
		case CTL_APPLY: {
			P_bias_data[PR_bias_element] += PR_top_data[PR_top_element];
			P_top_element++;
			P_channel_idx++;
			P_top_idx++;
			HtContinue(CTL_INCREMENT);
		}
		break;
		case CTL_INCREMENT: {
			if(P_top_channel_idx > channel_stride){
				P_top_channel_idx=0;
				P_bias_idx++;
				P_bias_element_idx++;
			}
			HtContinue(CTL_TOP_TEST);
		}
		break;
		case CTL_RTN: {
			BUSY_RETRY(SendReturnBusy_convbackbias());
			SendReturn_convbackbias();
		}
		break;
		default:
			assert(0);
		}
	}
}
