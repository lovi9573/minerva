/*
 * task.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: jlovitt
 */
#include <vector>
#include "device/task.h"

namespace minerva{

int Task::GetSerializedSize() const {
	int size = 0;
	size += sizeof(size_t);
	for(auto i : inputs){
		size += i.physical_data.GetSerializedSize();
	}
	size += sizeof(size_t);
	for(auto o : outputs){
		size += o.physical_data.GetSerializedSize();
	}
	size += op.GetSerializedSize();
	size += sizeof(uint64_t);
	return size;
}

int Task::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, inputs.size(), size_t)
	for(auto i : inputs){
		offset += i.physical_data.Serialize(buffer+offset);
	}
	SERIALIZE(buffer, offset, outputs.size(), size_t)
	for(auto o : outputs){
		offset += o.physical_data.Serialize(buffer+offset);
	}
	offset += op.Serialize(buffer+offset);
	SERIALIZE(buffer, offset, id, uint64_t)
	return offset;
}

Task& Task::DeSerialize(char* buffer, int* bytes) {
	int b = 0;
	int offset = 0;
	size_t n;
	Task *task = new Task();
	DESERIALIZE(buffer, offset, n, size_t)
	for(size_t i = 0; i < n; i++){
		task->inputs.emplace_back(TaskData(PhysicalData::DeSerialize(buffer+offset, &b),0));
		offset += b;
	}
	DESERIALIZE(buffer, offset, n, size_t)
	for(size_t i = 0; i < n; i++){
		task->outputs.emplace_back(TaskData(PhysicalData::DeSerialize(buffer+offset, &b),0));
		offset += b;
	}
    task->op = PhysicalOp::DeSerialize(buffer+offset, &b);
    offset += b;
    DESERIALIZE(buffer, offset, task->id, uint64_t)
    *bytes = offset;
    return *task;
}


} //end namespace minerva
