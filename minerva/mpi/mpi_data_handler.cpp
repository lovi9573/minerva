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
#include <chrono>
#include <iostream>


namespace minerva {

MpiDataHandler::MpiDataHandler(int r): rank_(r), next_response_buffer(nullptr) {
}

MpiDataHandler::~MpiDataHandler() {
}



//TODO(jesselovitt): Defer handling requests until our requests are filled IF we have a lower rank than requester.
void MpiDataHandler::Handle_Task_Data_Request(MPI_Status& status){
	std::cout << "[" << rank_ << "] Aquiring lock in order to handle data request" << "\n";
	std::unique_lock<std::mutex> lock(mpi_mutex_);
	int count ;
	MPI_Get_count(&status, MPI_BYTE, &count);
	char buffer[count];
	int offset = 0;
	uint64_t device_id;
	uint64_t data_id;
	uint64_t bytes;
	MPI_Recv(buffer, count, MPI_BYTE, status.MPI_SOURCE,MPI_TASK_DATA_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
	DESERIALIZE(buffer, offset, device_id, uint64_t)
	DESERIALIZE(buffer, offset, data_id, uint64_t)
	DESERIALIZE(buffer, offset, bytes, uint64_t)
	std::cout << "[" << rank_ << "] handling data request for data id: " << data_id << "\n";
	std::pair<Device::MemType, element_t*> devptr = MinervaSystem::Instance().GetPtr(device_id, data_id);
	//char outbuffer[bytes];
	//MinervaSystem::Instance().UniversalMemcpy(to, devptr, bytes);
/*	for(uint64_t i =0; i < bytes/sizeof(element_t); i ++){
		printf("%f, ", *(to.second+i));
	}*/

	//printf("[%d] Mpi data handler sending %lu element_t's over\n", MinervaSystem::Instance().rank(),(bytes/sizeof(elemnt_t)));
	MPI_Send(devptr.second, bytes, MPI_BYTE, status.MPI_SOURCE, MPI_TASK_DATA_RESPONSE, MPI_COMM_WORLD);
}

void MpiDataHandler::Request_Data(char* devbuffer, size_t bytes, int rank, uint64_t device_id, uint64_t data_id){
	{
		std::cout << "[" << rank_ << "] R-pre-request A " << data_id << "\n";
		std::unique_lock<std::mutex> lock(mpi_mutex_);
		std::cout << "[" << rank_ << "] R-post-request A " << data_id << "\n";
		while( next_response_buffer != nullptr){
			mpi_request_complete_.wait(lock);
			std::cout << "[" << rank_ << "] R-post-wake A " << data_id << "\n";
		}
		std::cout << "[" << rank_ << "] R-sending  data request for data id: " << data_id << "\n";
		int size = sizeof(size_t) + 2*sizeof(uint64_t);
		char msgbuffer[size];
		int offset = 0;
		SERIALIZE(msgbuffer, offset, device_id, uint64_t);
		SERIALIZE(msgbuffer, offset, data_id, uint64_t);
		SERIALIZE(msgbuffer, offset, bytes, size_t);
		next_response_buffer = devbuffer;
		next_data_id = data_id;
		MPI_Send(msgbuffer, size, MPI_BYTE, rank, MPI_TASK_DATA_REQUEST, MPI_COMM_WORLD);
		//printf("[%d] Mpi data handler recieving %lu element_t's\n", MinervaSystem::Instance().rank(),(bytes/sizeof(element_t)));
		//MPI_Recv(devbuffer, bytes, MPI_BYTE, rank, MPI_TASK_DATA_RESPONSE, MPI_COMM_WORLD, &status);

	//TODO(jesselovitt) We can't just let the thread out of this method.  It will continue because it thinks it has the data.
		while ( next_response_buffer != nullptr){
			std::cout << "[" << rank_ << "] R-pre-sleep B " << data_id << "\n";
			mpi_receive_complete_.wait(lock);
			std::cout << "[" << rank_ << "] R-post-wake B " << data_id << "\n";
		}
		std::cout << "[" << rank_ << "] R-post-wait B " << data_id << "\n";
	}
	mpi_request_complete_.notify_all();
}

void MpiDataHandler::Handle_Task_Data_Response(MPI_Status status){
	{
		std::cout << "[" << rank_ << "] pre-handle A " << next_data_id << "\n";
		std::unique_lock<std::mutex> lock(mpi_mutex_);
		std::cout << "[" << rank_ << "] post-handle A " << next_data_id << "\n";
		int bytes;
		MPI_Get_count(&status, MPI_BYTE, &bytes);
		MPI_Recv(next_response_buffer, bytes, MPI_BYTE, status.MPI_SOURCE, MPI_TASK_DATA_RESPONSE, MPI_COMM_WORLD, &status);
		std::cout << "[" << rank_ << "] received data from last request:  " << next_data_id  << "\n";
		next_response_buffer = nullptr;
		next_data_id = 0;
	}
	mpi_receive_complete_.notify_all();
}

} /* namespace minerva */

#endif // HAS_MPI
