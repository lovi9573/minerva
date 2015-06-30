/*
 * MpiDataHandler.cpp
 *
 *  Created on: Jun 23, 2015
 *      Author: jlovitt
 */

#ifdef HAS_MPI

#include "mpi/mpi_data_handler.h"
#include "mpi/mpi_common.h"
#include "common/common.h"
#include "device/device.h"
#include "system/minerva_system.h"

namespace minerva {

MpiDataHandler::MpiDataHandler() {
}

MpiDataHandler::~MpiDataHandler() {
}


	//TODO(jlovitt): This will be important when using async communication.
void MpiDataHandler::Handle_Task_Data(::MPI::Status& status){
	//int count = status.Get_count(MPI_TASKDATA);
	//MpiTaskData taskdata[count];
	//::MPI::COMM_WORLD.Recv(&taskdata, count, MPI_TASKDATA, status.Get_source(),MPI_TASK_DATA );
}

void MpiDataHandler::Handle_Task_Data_Request(::MPI::Status& status){
	int count = status.Get_count(MPI_BYTE);
	char buffer[count];
	int offset = 0;
	uint64_t device_id;
	uint64_t data_id;
	uint64_t bytes;
	::MPI::COMM_WORLD.Recv(buffer, count, MPI_BYTE, status.Get_source(),MPI_TASK_DATA_REQUEST );
	DESERIALIZE(buffer, offset, device_id, uint64_t)
	DESERIALIZE(buffer, offset, data_id, uint64_t)
	DESERIALIZE(buffer, offset, bytes, uint64_t)
	std::pair<Device::MemType, float*> devptr = MinervaSystem::Instance().GetPtr(device_id, data_id);
	//char outbuffer[bytes];
	//std::pair<Device::MemType, float*> to = std::pair<Device::MemType, float*>(Device::MemType::kCpu, (float*)outbuffer);
	//MinervaSystem::Instance().UniversalMemcpy(to, devptr, bytes);
/*	for(uint64_t i =0; i < bytes/sizeof(float); i ++){
		printf("%f, ", *(to.second+i));
	}*/

	//printf("[%d] Mpi data handler sending %lu floats over\n", MinervaSystem::Instance().rank(),(bytes/sizeof(float)));
	::MPI::COMM_WORLD.Send(devptr.second, bytes, MPI_BYTE, status.Get_source(), MPI_TASK_DATA_RESPONSE);
}

void MpiDataHandler::Request_Data(char* devbuffer, size_t bytes, int rank, uint64_t device_id, uint64_t data_id){
	int size = sizeof(size_t) + 2*sizeof(uint64_t);
	char msgbuffer[size];
	int offset = 0;
	SERIALIZE(msgbuffer, offset, device_id, uint64_t);
	SERIALIZE(msgbuffer, offset, data_id, uint64_t);
	SERIALIZE(msgbuffer, offset, bytes, size_t);
	::MPI::COMM_WORLD.Send(msgbuffer, size, MPI_BYTE, rank, MPI_TASK_DATA_REQUEST);
	//printf("[%d] Mpi data handler recieving %lu floats\n", MinervaSystem::Instance().rank(),(bytes/sizeof(float)));
	::MPI::COMM_WORLD.Recv(devbuffer, bytes, MPI_BYTE, rank, MPI_TASK_DATA_RESPONSE);
/*
	for(uint64_t i =0; i < bytes/sizeof(float); i ++){
		printf("%f, ", *(  ((float*)buffer)+i        ));
	}*/
}

} /* namespace minerva */

#endif // HAS_MPI
