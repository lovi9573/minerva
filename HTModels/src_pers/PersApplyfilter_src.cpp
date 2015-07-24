/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersApplyfilter.h"

#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }



void
CPersApplyfilter::PersApplyfilter()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case CONV_INIT: {
			//BUSY_RETRY(ReadMemBusy());
			//printf("filter %d,%d,%d\n",PR_filter_fIdx,PR_img_xIdx,PR_img_yIdx);
			//Initialize and Get the filter data.
			P_cIdx = 0;
			P_xIdx = 0;
			P_yIdx = 0;
			P_accum=0;
			//ReadMem_filter(SR_filterAddr,0, 18); //18 is #elements.  Filter is 2x3x3=> 18
			HtContinue(CONV_LOOP_TOP);
		}
		break;
		case CONV_LOOP_TOP: {
			//printf("loop top\n");
			HtContinue(CONV_LD_IMG_SAMPLE);
		}
		break;
		case CONV_LD_IMG_SAMPLE: {
			BUSY_RETRY(ReadMemBusy());
			//printf("img read\n");
			ReadMem_img_val(PR_imgAddr+2*(PR_yIdx*SR_img_dim*SR_img_channels+
										  PR_xIdx*SR_img_channels+
										  PR_cIdx)); //2* for 16-bit align
			ReadMemPause(CONV_LD_FILTER_SAMPLE);
		}
		break;
		case CONV_LD_FILTER_SAMPLE: {
			BUSY_RETRY(ReadMemBusy());
			//printf("filter read address set to %d\n",PR_filter_yIdx*SR_filter_dim*SR_img_channels+PR_filter_xIdx*SR_img_channels+PR_filter_cIdx);
			ReadMem_filter_val(PR_filterAddr+2*(PR_yIdx*SR_filter_dim*SR_img_channels+
											    PR_xIdx*SR_img_channels+
											    PR_cIdx));
			ReadMemPause(CONV_APPLY);
		}
		break;
		case CONV_APPLY: {
			//printf("filter read\n");
			int32_t result = (( ((int32_t)PR_img_val) * ((int32_t)PR_filter_val) ) >> SR_fractionW) +(int32_t)PR_accum;

			if(result >= 0x00007FFFL){
				P_accum = 0x7FFF;
			}
			//printf("result %d vs %d\n",result,-0x00007FFF);
			else if(result <= (int16_t)0x8000){
				P_accum = (int16_t)0x8000;
			}
			else{

				P_accum = (int16_t)result;
			}
			P_cIdx++;
			printf("accum  %d * %d  >> %d = %d\n",PR_img_val,PR_filter_val, SR_fractionW,P_accum);
			HtContinue(CONV_LOOP_BRANCH);
		}
		break;
		case CONV_LOOP_BRANCH: {
			if(P_cIdx >= SR_img_channels){
				P_cIdx -= SR_img_channels;
				P_xIdx++;
				if(P_xIdx >= SR_filter_dim){
					P_xIdx -= SR_filter_dim;
					P_yIdx++;
				}
			}
			//Still need to reduce the channel offset
			if(PR_cIdx >= SR_img_channels){
				HtContinue(CONV_LOOP_BRANCH);
			}
			//Still need to process
			else if(PR_yIdx < SR_filter_dim){
				HtContinue(CONV_LOOP_TOP);
			}
			//Done
			else{
				HtContinue(CONV_RTN);
			}
		}
		break;
		case CONV_RTN: {
			BUSY_RETRY(SendReturnBusy_applyfilter());
			//printf("Accum: %d\n",PR_accum);
			SendReturn_applyfilter(PR_out_index,PR_accum);
		}
		break;
		default:
			assert(0);
		}
	}
}
