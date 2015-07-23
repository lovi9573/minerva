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
			P_vIdx = 0;
			P_valN = 1;
			//ReadMem_filter(SR_filterAddr,0, 18); //18 is #elements.  Filter is 2x3x3=> 18
			HtContinue(CONV_LOOP_TOP);
		}
		break;
		case CONV_LD_LOOP_TOP: {
			//printf("loop top\n");
			if(PR_valN == 1){
				P_imgOffset = 2*(PR_yIdx*SR_img_dim*SR_img_channels+
						 PR_xidx*SR_img_channels+
						 PR_cIdx);
				P_filterOffset = 2*(PR_yIdx*SR_filter_dim*SR_img_channels+
						    PR_xIdx*SR_img_channels+
						    PR_cIdx);
			}
			//Ready to load 64 contiguous bits
			if(PR_valN == 4){
				HtContinue(CONV_LD_IMG_SAMPLE);
			}
			//Check for overflow
			else{ 
				if(P_cIdx >= SR_img_channels){
					P_cIdx -= SR_img_channels;
					P_xIdx++;
					if(P_xIdx >= SR_filter_dim){
						P_xIdx -= SR_filter_dim;
						P_yIdx++; //Not contiguous in image space.
						HtContinue(CONV_LD_IMG_SAMPLE);
						break;
					}
				}
				//Still need to process
				if(PR_yIdx < SR_filter_dim){
					//Count another contigous value position
					P_cIdx++;
					P_vIdx++;
					P_valN++;		
					HtContinue(CONV_LD_LOOP_TOP);
				}
				else{
					HtContinue(CONV_RTN);
				}
			}
		}
		break;
		case CONV_LD_IMG_SAMPLE: {
			BUSY_RETRY(ReadMemBusy());
			//printf("img read\n");
			ReadMem_img_val(PR_imgAddr+PR_imgOffset;
			ReadMemPause(CONV_LD_FILTER_SAMPLE);
		}
		break;
		case CONV_LD_FILTER_SAMPLE: {
			BUSY_RETRY(ReadMemBusy());
			//printf("filter read address set to %d\n",PR_filter_yIdx*SR_filter_dim*SR_img_channels+PR_filter_xIdx*SR_img_channels+PR_filter_cIdx);
			ReadMem_filter_val(PR_filterAddr+PR_filterOffset;
			ReadMemPause(CONV_APPLY);
		}
		break;
		case CONV_APPLY: {
			//printf("filter read\n");
			P_accum+= (int16_t)(( ((int32_t)PR_img_val[PR_valIdx]) * ((int32_t)PR_filter_val[PR_valIdx]) ) >> SR_fractionW); // for fixed point>>16);
			P_valIdx++;
			//printf("accum  %d * %d  >> %d = %d\n",PR_img_val,PR_filter_val, GR_fractionW,P_accum);
			HtContinue(CONV_LOOP_BRANCH);
		}
		break;
		case CONV_LOOP_BRANCH: {
			if(PR_valIdx == PR_valN){
				P_valN = 1;
				P_valIdx = 0;
				P_cIdx++;
			}
			else{
				HtContinue(CONV_APPLY);
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
