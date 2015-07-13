/*
 * relu_forward.c
 *
 *  Created on: Jul 7, 2015
 *      Author: jlovitt
 */


#include "Ht.h"
using namespace Ht;


void CHtModelAuUnit::UnitThread()
{
	uint64_t *op1Addr = NULL, *resAddr = NULL;
	uint64_t vecLen = 0;

	do {
		uint8_t msgType;
		uint64_t msgData;
		if (RecvHostMsg(msgType, msgData)) {
			switch (msgType) {
			case IN_ADDR: op1Addr = (uint64_t *)msgData; break;
			case OUT_ADDR: resAddr = (uint64_t *)msgData; break;
			case VEC_LEN:  vecLen = msgData; break;
			default: assert(0);
			}
		}

		uint32_t vecIdx, vecStride;
		if (RecvCall_htmain(vecIdx, vecStride)) {
			while (vecIdx < vecLen) {
				uint64_t op1 = *(op1Addr + vecIdx);
				uint64_t res = op1 ;
				if(op1 <=0){
					res = 0;
				}
				*(resAddr + vecIdx) = res;
				vecIdx += vecStride;
			}
			while (!SendReturn_htmain());
		}
	} while (!RecvHostHalt());
	// The CHtHif class destructor issues a "Halt" message
	// to the units to terminate execution. The do {} while(),
	// mimics the behavior of the functionality provided by
	// the HT infrastructure.
}
