/*
 * relu_forward_pers.cpp
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */

#include "Ht.h"
#include "PersCluster.h"

#define BUSY_RETRY(b) { if (b) { HtRetry(); break; } }



void
CPersCluster::PersCluster()
{
	if (PR_htValid) {
		switch (PR_htInst) {
		case SG_FILTER_GROUP_INIT: {
			//printf("scattergatherfiltergroup apply %lu to %lu => %lu\n",PR_filter_Addr1,PR_img_Addr1,PR_out_Addr);
			//Initialize and Get the filter data.
			P_activation_group = 0;
			P_idx = 0;
			//ReadMem_filter(SR_filterAddr,0, 18); //18 is #elements.  Filter is 2x3x3=> 18
			HtContinue(SG_FILTER_GROUP_DISPATCH);
		}
		break;
		case SG_FILTER_GROUP_DISPATCH: {
			if(PR_idx == 0){
				BUSY_RETRY(SendCallBusy_applyfilter());
				SendCallFork_applyfilter(SG_FILTER_GROUP_MERGE, PR_idx, PR_img_Addr1, PR_filter_Addr1);
				P_idx = PR_idx + 1;
				HtContinue(SG_FILTER_GROUP_DISPATCH);
			}
			else if(PR_idx == 1){
				BUSY_RETRY(SendCallBusy_applyfilter());
				SendCallFork_applyfilter(SG_FILTER_GROUP_MERGE, PR_idx, PR_img_Addr2, PR_filter_Addr2);
				P_idx = PR_idx + 1;
				HtContinue(SG_FILTER_GROUP_DISPATCH);
				}
			else if(PR_idx == 2){
				BUSY_RETRY(SendCallBusy_applyfilter());
				SendCallFork_applyfilter(SG_FILTER_GROUP_MERGE, PR_idx, PR_img_Addr3, PR_filter_Addr3);
				P_idx = PR_idx + 1;
				HtContinue(SG_FILTER_GROUP_DISPATCH);
				}
			else if(PR_idx == 3){
				BUSY_RETRY(SendCallBusy_applyfilter());
				SendCallFork_applyfilter(SG_FILTER_GROUP_MERGE, PR_idx, PR_img_Addr4, PR_filter_Addr4);
				P_idx = PR_idx + 1;
				HtContinue(SG_FILTER_GROUP_DISPATCH);
				}
			else{
				RecvReturnPause_applyfilter(SG_FILTER_GROUP_WRITE);
			}
		}
		break;
		case SG_FILTER_GROUP_MERGE: {
			RecvReturnJoin_applyfilter();
			int shift = 16*PR_out_index;
			uint64_t mask = 0x000000000000FFFFULL << shift;
			//printf("accum: %d << %d mask %lx\n",PR_accum,shift,mask);
			uint64_t shifted_accum = ((uint64_t)PR_accum) << shift ;
			//printf("shifted accum: %lx\n",shifted_accum);
			P_activation_group = (PR_activation_group & ~mask) | shifted_accum;
		}
		break;
		case SG_FILTER_GROUP_WRITE: {
			BUSY_RETRY(WriteMemBusy());
			//printf("write activation: %lx \n",PR_activation_group);
			WriteMem(PR_out_Addr, P_activation_group);
			WriteMemPause(SG_FILTER_GROUP_RTN);
		}
		break;
		case SG_FILTER_GROUP_RTN: {
			BUSY_RETRY(SendReturnBusy_cluster());
			SendReturn_cluster();
		}
		break;
		default:
			assert(0);
		}
	}
}
