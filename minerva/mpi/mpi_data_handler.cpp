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
#include "common/cuda_utils.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cuda.h>
#include <cublas_v2.h>
#endif


namespace minerva {

MpiDataHandler::MpiDataHandler(int r): rank_(r), id_(r), send_complete_(0), send_request_valid_(false) {
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
	uint64_t ticker = 0;
	while (!term){
		if(++ticker%100000000 == 0){
			MPILOG << "["<<rank_<<"](" << ticker << ")\n";

		}
		MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &pending_message, &status);
		if(pending_message){
			MPI_Get_count(&status, MPI_BYTE, &count);
			char* buffer = new char[count];
			MPI_Recv(buffer, count, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
			int offset = 0;
			DESERIALIZE(buffer, offset, id, uint64_t)
			MPILOG << "[" << rank_  << "]<= {mainloop} ==== Handling message tag " << status.MPI_TAG << " id "<< id <<". ====\n";
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
		if(send_request_valid_){
			MPI_Test(&send_request_, &send_complete_, MPI_STATUS_IGNORE);
			if(send_complete_){
				std::unique_lock<std::mutex> lock(send_mutex_);
				if(send_queue_.size() > 0){
					send_complete_ = 0;
					SendItem& s = send_queue_.front();
					char* buffer = new char[s.size + sizeof(uint64_t)];
					int offset = 0;
					SERIALIZE(buffer, offset, s.id, uint64_t)
					#ifdef HAS_CUDA
					  	  CUDA_CALL(cudaMemcpy(buffer+offset, s.buffer,  s.size, cudaMemcpyDefault));
					#else
						std::memcpy(buffer+offset, s.buffer, s.size);
					#endif
					MPILOG << "[" << rank_  << "]=> {mainloop} ==== Sending message tag " << s.tag << " id "<< s.id <<". ====\n";
					MPI_Isend(buffer, s.size+offset, MPI_BYTE, s.dest_rank, s.tag, MPI_COMM_WORLD, &send_request_);
					send_queue_.pop();
					//delete[] buffer;
				}
			}
		}else{
			if(send_queue_.size() > 0){
				send_complete_ = 0;
				SendItem& s = send_queue_.front();
				char* buffer = new char[s.size + sizeof(uint64_t)];
				int offset = 0;
				SERIALIZE(buffer, offset, s.id, uint64_t)
				#ifdef HAS_CUDA
					  CUDA_CALL(cudaMemcpy(buffer+offset, s.buffer,  s.size, cudaMemcpyDefault));
				#else
					std::memcpy(buffer+offset, s.buffer, s.size);
				#endif
				MPI_Isend(buffer, s.size+offset, MPI_BYTE, s.dest_rank, s.tag, MPI_COMM_WORLD, &send_request_);
				MPILOG << "[" << rank_  << "]=> {mainloop} ==== Sending message tag " << s.tag << " id "<< s.id <<". ====\n";
				send_queue_.pop();
				send_request_valid_ = true;
//				delete[] buffer;
			}
		}
	}
	MPI_Finalize();
}

//TODO(JesseLovitt): There is still a deadlock here for some reason.
void MpiDataHandler::Request_Data(char* devbuffer, size_t bytes, int rank, uint64_t device_id, uint64_t data_id){
		//Serialize message
		int size = sizeof(size_t) + 2*sizeof(uint64_t);
		char msgbuffer[size];
		int offset = 0;
		SERIALIZE(msgbuffer, offset, device_id, uint64_t);
		SERIALIZE(msgbuffer, offset, data_id, uint64_t);
		SERIALIZE(msgbuffer, offset, bytes, size_t);
		MPILOG_DATA << "[" << rank_  << "]=> {" << std::this_thread::get_id() << "} Requesting "<< bytes <<" Bytes, id "<< data_id <<" from rank "<< rank << ".\n";
		RecvItem r_item(devbuffer);
		uint64_t mpi_id = Get_Mpi_Id();
		{
			std::unique_lock<std::mutex> lock(recv_mutex_);
			recv_buffer_.insert( std::pair<uint64_t,RecvItem>(mpi_id, r_item) );
		}
		//Get a unique id and queue the message up.
		Send(mpi_id, msgbuffer,size,rank,MPI_TASK_DATA_REQUEST);

		//prepare receive structure and wait for response.
		Wait_For_Recv(mpi_id, devbuffer);
}

int MpiServer::rank(){
	return rank_;
}

/*
 *  ========= Protected and Private =========
 */

//TODO(JesseLovitt): Rework this section to be more interface-like for someone who wants to send - wait (i.e. they should not have to generate their own mpi_id)
uint64_t MpiDataHandler::Get_Mpi_Id(){
	std::unique_lock<std::mutex> lock(id_mutex_);
	id_ += id_stride_;
	return id_;
}


uint64_t MpiDataHandler::Send(char* msgbuffer, int size, int rank, int tag){
	uint64_t mpi_id = Get_Mpi_Id();
	return Send(mpi_id, msgbuffer, size, rank, tag);
}

uint64_t MpiDataHandler::Send(uint64_t mpi_id, char* msgbuffer, int size, int rank, int tag){
	//Get a unique id and queue the message up.
	MPILOG << "[" << rank_  << "]=> {" << std::this_thread::get_id() << "} Queuing "<< size <<" Byte message, type " << tag << " for rank "<< rank << ".\n";
	SendItem item(mpi_id,msgbuffer,size,rank,tag);
	{
		std::unique_lock<std::mutex> lock(send_mutex_);
		send_queue_.push(item);
	}
	return mpi_id;
}

void MpiDataHandler::Wait_For_Recv(uint64_t mpi_id, char* buffer){
	std::unique_lock<std::mutex> lock(recv_mutex_);
	MPILOG_DATA << "[" << rank_  << "]=> {" << std::this_thread::get_id() << "} Waiting to receive message mpi_id "<< mpi_id << ".\n";
	while ( !recv_buffer_.at(mpi_id).ready ){
		recv_complete_.wait(lock);
		MPILOG_DATA << "[" << rank_  << "]=> {" << std::this_thread::get_id() << "} wait on message mpi_id "<< mpi_id << " status " << recv_buffer_.at(mpi_id).ready << ".\n";
	}
	recv_buffer_.erase(mpi_id);
	MPILOG_DATA << "[" << rank_  << "]=> {" << std::this_thread::get_id() << "} Received message mpi_id "<< mpi_id << ".\n";
}


/*
 *  ======== Receive Handlers =========
 */

void MpiDataHandler::Handle_Task_Data_Request(uint64_t mpi_id, char* buffer, size_t size, int rank){

	//Get and deserialize message.
	int offset = 0;
	uint64_t device_id;
	uint64_t data_id;
	uint64_t bytes;
	DESERIALIZE(buffer, offset, device_id, uint64_t)
	DESERIALIZE(buffer, offset, data_id, uint64_t)
	DESERIALIZE(buffer, offset, bytes, uint64_t)
	MPILOG_DATA << "[" << rank_ << "]<= {mainloop} handling data request for data id: " << data_id << " from device id: " << device_id << "\n";
	std::pair<Device::MemType, element_t*> devptr = MinervaSystem::Instance().GetPtr(device_id, data_id);
	Send(mpi_id,reinterpret_cast<char*>(devptr.second), bytes, rank, MPI_TASK_DATA_RESPONSE);
}


void MpiDataHandler::Handle_Task_Data_Response(uint64_t id, char* buffer, size_t size, int rank){
		std::unique_lock<std::mutex> lock(recv_mutex_);
		MPILOG_DATA << "[" << rank_ << "]<= {mainloop} handling data response for data, mpi_id: " << id << "\n";
		if(recv_buffer_.find(id) != recv_buffer_.end()){
			recv_buffer_.at(id).ready = 1;
			std::memcpy(recv_buffer_.at(id).buffer, buffer, size);
		}
		recv_complete_.notify_all();
}

void MpiDataHandler::Default_Handler(uint64_t id, char* buffer, size_t size, int rank, int tag){
	return;
}

} /* namespace minerva */

#endif // HAS_MPI
