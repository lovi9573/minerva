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

MpiDataHandler::MpiDataHandler(int r): rank_(r), id_(r), send_complete_(1) {
	std::unique_lock<std::mutex> lock(id_mutex_);
	MPI_Comm_size(MPI_COMM_WORLD, &id_stride_);
}

MpiDataHandler::~MpiDataHandler() {
}

void MpiDataHandler::MainLoop(){
	bool term = false;
	int pending_message = 0;
	MPI_Status status;
	int count ;
	uint64_t id;
	bool notify = false;
	//MPI_Status st;
	while (!term){
		MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &pending_message, &status);
		if(pending_message){
			DLOG(INFO) << "[" << rank_  << "] Handling message" << status.MPI_TAG << ".\n";
			MPI_Get_count(&status, MPI_BYTE, &count);
			char* buffer = new char[count];
			MPI_Recv(buffer, count, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
			int offset = 0;
			DESERIALIZE(buffer, id, offset, uint64_t)
			char* message = buffer + offset;
			count = count - offset;
			switch(status.MPI_TAG){
				case MPI_TASK_DATA_RESPONSE:
					Handle_Task_Data_Response(id, message, count, status.MPI_SOURCE);
					break;
				case MPI_TASK_DATA_REQUEST:
					Handle_Task_Data_Request(id, message, count, status.MPI_SOURCE);
					break;
				case MPI_TERMINATE:
					term = true;
					break;
				default:
					Default_Handler(id, message, count, status.MPI_SOURCE, status.MPI_TAG);
			}
			pending_message = 0;
			delete[] buffer;
			{
				std::unique_lock<std::mutex> lock(recv_mutex_);
				if(recv_buffer_.find(id) != recv_buffer_.end()){
					recv_buffer_.at(id).ready = 1;
					notify = true;
				}
			}
			if(notify){
				recv_complete_.notify_all();
				notify = false;
			}
		}
		MPI_Test(&send_request_, &send_complete_, MPI_STATUS_IGNORE);
		if(send_complete_){
			std::unique_lock<std::mutex> lock(send_mutex_);
			if(send_queue_.size() > 0){
				send_complete_ = 0;
				SendItem& s = send_queue_.front();
				MPI_Isend(s.buffer, s.size, MPI_BYTE, s.dest_rank, s.tag, MPI_COMM_WORLD, &send_request_);
				send_queue_.pop();
			}
		}
	}
	MPI_Finalize();
}

void MpiDataHandler::Request_Data(char* devbuffer, size_t bytes, int rank, uint64_t device_id, uint64_t data_id){
		//Serialize message
		int size = sizeof(size_t) + 2*sizeof(uint64_t);
		char msgbuffer[size];
		int offset = 0;
		SERIALIZE(msgbuffer, offset, device_id, uint64_t);
		SERIALIZE(msgbuffer, offset, data_id, uint64_t);
		SERIALIZE(msgbuffer, offset, bytes, size_t);

		//Get a unique id and queue the message up.
		uint64_t mpi_id = Send(msgbuffer,size,rank,MPI_TASK_DATA_REQUEST);

		//prepare receive structure and wait for response.
		Wait_For_Recv(mpi_id, devbuffer);
}

int MpiServer::rank(){
	return rank_;
}

/*
 *  ========= Protected and Private =========
 */


uint64_t MpiDataHandler::Get_Mpi_Id(){
	std::unique_lock<std::mutex> lock(id_mutex_);
	id_ += id_stride_;
	return id_;
}

uint64_t MpiDataHandler::Send(char* msgbuffer, int size, int rank, int tag){
	//Get a unique id and queue the message up.
	uint64_t mpi_id = Get_Mpi_Id();
	SendItem item(mpi_id,msgbuffer,size,rank,MPI_TASK_DATA_REQUEST);
	{
		std::unique_lock<std::mutex> lock(send_mutex_);
		send_queue_.push(item);
	}
	return mpi_id;
}

void MpiDataHandler::Wait_For_Recv(uint64_t mpi_id, char* buffer){
	std::unique_lock<std::mutex> lock(recv_mutex_);
	RecvItem r_item(buffer);
	recv_buffer_.insert( std::pair<uint64_t,RecvItem>(mpi_id, r_item) );
	while ( !recv_buffer_.at(mpi_id).ready ){
		recv_complete_.wait(lock);
	}
	recv_buffer_.erase(mpi_id);
}


/*
 *  ======== Receive Handlers =========
 */

void MpiDataHandler::Handle_Task_Data_Request(uint64_t id, char* buffer, size_t size, int rank){

	//Get and deserialize message.
	int offset = 0;
	uint64_t device_id;
	uint64_t data_id;
	uint64_t bytes;
	DESERIALIZE(buffer, offset, device_id, uint64_t)
	DESERIALIZE(buffer, offset, data_id, uint64_t)
	DESERIALIZE(buffer, offset, bytes, uint64_t)
	//std::cout << "[" << rank_ << "] handling data request for data id: " << data_id << "\n";
	std::pair<Device::MemType, element_t*> devptr = MinervaSystem::Instance().GetPtr(device_id, data_id);
	Send(reinterpret_cast<char*>(devptr.second), bytes, rank, MPI_TASK_DATA_RESPONSE);
}


void MpiDataHandler::Handle_Task_Data_Response(uint64_t id, char* buffer, size_t size, int rank){
		std::unique_lock<std::mutex> lock(recv_mutex_);
		if(recv_buffer_.find(id) != recv_buffer_.end()){
			recv_buffer_.at(id).ready = 1;
			std::memcpy(recv_buffer_.at(id).buffer, buffer, size);
		}
}

void MpiDataHandler::Default_Handler(uint64_t id, char* buffer, size_t size, int rank, int tag){
	return;
}

} /* namespace minerva */

#endif // HAS_MPI
