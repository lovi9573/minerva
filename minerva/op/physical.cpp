/*
 * physical.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: jlovitt
 */

#include "op/physical.h"
#include "op/compute_fn.h"

namespace minerva {

int PhysicalData::GetSerializedSize() const {
	return size.GetSerializedSize()+2*sizeof(uint64_t)
#ifdef HAS_MPI
			+sizeof(int)
#endif
;
}

int PhysicalData::Serialize(char* buffer) const {
	int offset = 0;
	offset += size.Serialize(buffer);
	SERIALIZE(buffer, offset, device_id, uint64_t)
	SERIALIZE(buffer, offset, data_id, uint64_t)
	//*(buffer+offset) = extern_rc;
	//offset += sizeof(int);
#ifdef HAS_MPI
	SERIALIZE(buffer, offset, rank, int)
#endif
	return offset;
}

PhysicalData& PhysicalData::DeSerialize(char* buffer,int* offset){

	Scale& s = Scale::DeSerialize(buffer, offset);
	uint64_t d = *((uint64_t*)(buffer+*offset));
	*offset += sizeof(uint64_t);
	uint64_t id = *((uint64_t*)(buffer+*offset));
	*offset += sizeof(uint64_t);
	//int extern_rc = *((int*)(buffer+*offset));
	//*offset += sizeof(int);
#ifdef HAS_MPI
	int r = *((int*)(buffer+*offset));
	*offset += sizeof(int);
	return *(new PhysicalData(s,r,d,id));
#else
	return *(new PhysicalData(s,d,id));
#endif
}

int PhysicalOp::GetSerializedSize() const {
	return compute_fn->GetSerializedSize()+sizeof(uint64_t);
}

int PhysicalOp::Serialize(char* buffer) const {
	int offset = 0;
	offset += compute_fn->Serialize(buffer);
	SERIALIZE(buffer, offset, device_id, uint64_t)
	return offset;
}

PhysicalOp& PhysicalOp::DeSerialize(char* buffer, int* bytes){
	int offset = 0;
	int b = 0;
	PhysicalOp *op = new PhysicalOp();
	op->compute_fn = ComputeFn::DeSerialize(buffer,&b);
	offset += b;
	op->device_id = *((uint64_t*)(buffer+offset));
	offset += sizeof(uint64_t);
	*bytes = offset;
	return *op;
}


}// end namespace minerva
