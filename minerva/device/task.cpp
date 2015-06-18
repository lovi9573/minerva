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
	*(buffer+offset) = inputs.size();
	offset += sizeof(size_t);
	for(auto i : inputs){
		offset += i.physical_data.Serialize(buffer+offset);
	}
	*(buffer+offset) = outputs.size();
	offset += sizeof(size_t);
	for(auto o : outputs){
		offset += o.physical_data.Serialize(buffer+offset);
	}
	offset += op.Serialize(buffer+offset);
	*(buffer+offset) = id;
	offset += sizeof(uint64_t);
	return offset;
}

Task& Task::DeSerialize(char* buffer, int* offset) {
	Task *task = new Task();
	size_t n = *(size_t*)(buffer+*offset);
	*offset += sizeof(size_t);
	for(size_t i = 0; i < n; i++){
		task->inputs.emplace_back(TaskData(PhysicalData::DeSerialize(buffer, offset),0));
	}
	n = *(size_t*)(buffer+*offset);
	*offset += sizeof(size_t);
	for(size_t i = 0; i < n; i++){
		task->outputs.emplace_back(TaskData(PhysicalData::DeSerialize(buffer, offset),0));
	}
    task->op = PhysicalOp::DeSerialize(buffer,offset);
    task->id = *((uint64_t*)(buffer+*offset));
    *offset += sizeof(uint64_t);
    return *task;
}


} //end namespace minerva
