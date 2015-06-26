/*
 * MpiDataHandler.cpp
 *
 *  Created on: Jun 23, 2015
 *      Author: jlovitt
 */

#include "mpi/mpi_data_handler.h"
#include "mpi/mpi_common.h"
#include "common/common.h"
#include "device/device.h"
#include "system/minerva_system.h"

namespace minerva {

MpiDataHandler::MpiDataHandler() {
	// TODO Auto-generated constructor stub

}

MpiDataHandler::~MpiDataHandler() {
	// TODO Auto-generated destructor stub
}


void MpiDataHandler::Handle_Task_Data(::MPI::Status& status){
	//int count = status.Get_count(MPI_TASKDATA);
	//MpiTaskData taskdata[count];
	//::MPI::COMM_WORLD.Recv(&taskdata, count, MPI_TASKDATA, status.Get_source(),MPI_TASK_DATA );
	//TODO: 4 Put this task data somewhere...
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
	char outbuffer[bytes];
	std::pair<Device::MemType, float*> devptr = MinervaSystem::Instance().GetPtr(device_id, data_id);
	std::pair<Device::MemType, float*> to = std::pair<Device::MemType, float*>(Device::MemType::kCpu, (float*)outbuffer);
	//printf("Mpi data handler copying %lu floats over\n",(bytes/sizeof(float)));
	MinervaSystem::Instance().UniversalMemcpy(to, devptr, bytes);
/*	for(uint64_t i =0; i < bytes/sizeof(float); i ++){
		printf("%f, ", *(to.second+i));
	}*/
	::MPI::COMM_WORLD.Send(outbuffer, bytes, MPI_BYTE, status.Get_source(), MPI_TASK_DATA_RESPONSE);
}

void MpiDataHandler::Request_Data(char* buffer, size_t bytes, int rank, uint64_t device_id, uint64_t data_id){
	int size = sizeof(size_t) + 2*sizeof(uint64_t);
	char msgbuffer[size];
	int offset = 0;
	SERIALIZE(msgbuffer, offset, device_id, uint64_t);
	SERIALIZE(msgbuffer, offset, data_id, uint64_t);
	SERIALIZE(msgbuffer, offset, bytes, size_t);
	::MPI::COMM_WORLD.Send(msgbuffer, size, MPI_BYTE, rank, MPI_TASK_DATA_REQUEST);
	::MPI::COMM_WORLD.Recv(buffer, bytes, MPI_BYTE, rank, MPI_TASK_DATA_RESPONSE);
/*	printf("Mpi data handler recieving %lu floats\n",(bytes/sizeof(float)));
	for(uint64_t i =0; i < bytes/sizeof(float); i ++){
		printf("%f, ", *(  ((float*)buffer)+i        ));
	}*/
}

} /* namespace minerva */
