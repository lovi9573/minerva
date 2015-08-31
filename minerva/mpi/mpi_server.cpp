/*
 * serialize.h
 *
 *  Created on: Jun 4, 2015
 *      Author: jlovitt
 */

#ifdef HAS_MPI


#include <mpi.h>
#include "mpi_server.h"


using namespace std;

namespace minerva {


MPI_Datatype MPI_TASKDATA;

MpiServer::MpiServer(): MpiDataHandler(0){

}

void MpiServer::init(){
//	MPI_Init(0,NULL);
//	rank_ = ::MPI::COMM_WORLD.Get_rank();
	//_pendingTasks = ConcurrentUnorderedSet<uint64_t>();
}

void MpiServer::MainLoop(){
	int pending_message = 0;
	MPI_Status status;
	//MPI_Status st;
	while (listen_){
		{
			std::unique_lock<std::mutex> lock(mpi_mutex_);
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &pending_message, &status);
		}
		if(pending_message){
			DLOG(INFO) << "[" << rank_ << "] Handling message " << status.MPI_TAG ;
			switch(status.MPI_TAG){
			case MPI_TASK_DATA_REQUEST:
				Handle_Task_Data_Request(status);
				break;
			case MPI_TASK_DATA_RESPONSE:
				Handle_Task_Data_Response(status);
				break;
			case MPI_FINALIZE_TASK:
				Handle_Finalize_Task(status);
				break;
			default:
				Discard(status);
				break;
			}
			pending_message = 0;
		}
	}
}


int MpiServer::rank(){
	return rank_;
}

int MpiServer::GetMpiNodeCount(){
	int size = 0;
	std::unique_lock<std::mutex> lock(mpi_mutex_);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;
}

int MpiServer::GetMpiDeviceCount(int rank){
	int size = GetMpiNodeCount();
	if (rank >= size || rank == 0){
		return 0;
	}
	DLOG(INFO) << "Device count requested from rank #" << rank;
	int count;
	std::unique_lock<std::mutex> lock(mpi_mutex_);
	MPI_Send(0,0,MPI_INT, rank,MPI_DEVICE_COUNT, MPI_COMM_WORLD);
	MPI_Recv((void*)&count, 1, MPI_INT, rank,MPI_DEVICE_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	return count;
}

void MpiServer::CreateMpiDevice(int rank, int id, uint64_t device_id){
	DLOG(INFO) << "Creating device #" << id << " at rank "<< rank <<", uid " << device_id << "\n";
	int size = sizeof(int)+sizeof(uint64_t);
	char buffer[size];
	int offset = 0;
	SERIALIZE(buffer, offset, id, int)
	SERIALIZE(buffer, offset, device_id, uint64_t)
	std::unique_lock<std::mutex> lock(mpi_mutex_);
	MPI_Send(buffer,size, MPI_BYTE,rank,MPI_CREATE_DEVICE, MPI_COMM_WORLD);
}


void MpiServer::MPI_Send_task(const Task& task,const Context& ctx ){
	DLOG(INFO) << "Task #"<< task.id << " being sent to rank #" << ctx.rank;
	size_t bufsize = task.GetSerializedSize();
	char* buffer = new char[bufsize];
	size_t usedbytes = task.Serialize(buffer);
	CHECK_EQ(bufsize, usedbytes);
	{
		std::unique_lock<std::mutex> lock(mpi_mutex_);
		MPI_Request request;
		MPI_Isend(buffer, bufsize, MPI_BYTE, ctx.rank, MPI_TASK, MPI_COMM_WORLD, &request);
	}
	delete[] buffer;
	{
		std::unique_lock<std::mutex> lock(task_complete_mutex_);
		pending_tasks_.Insert(task.id);
	}
}

void MpiServer::Wait_On_Task(uint64_t task_id){
	std::unique_lock<std::mutex> lock(task_complete_mutex_);
	while(pending_tasks_.Count(task_id) > 0){
		task_complete_condition_.wait(lock);
	}
}

/**
 *  Signal Received From worker Rank that a task is complete
 */
void MpiServer::Handle_Finalize_Task(MPI_Status status){
	int count;
	uint64_t task_id;
	{
		std::unique_lock<std::mutex> lock(mpi_mutex_);
		MPI_Get_count(&status, MPI_BYTE, &count);
		//char buffer[count];
		MPI_Recv(&task_id, count, MPI_CHAR, status.MPI_SOURCE, MPI_FINALIZE_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//printf("notified of finalization of task %lu\n",task_id);
	}
	{
		std::unique_lock<std::mutex> task_lock(task_complete_mutex_);
		pending_tasks_.Erase(task_id);
	}
	task_complete_condition_.notify_all();
}



void MpiServer::Free_Data(int rank, uint64_t data_id){
	std::unique_lock<std::mutex> lock(mpi_mutex_);
	char buffer[sizeof(uint64_t)];
	int offset = 0;
	SERIALIZE(buffer, offset, data_id, uint64_t)
	//TODO(jesselovitt) This is not guaranteed to work.  This Isend should have a test before ANY other send's occur.
	MPI_Request request;
	MPI_Isend(buffer, offset, MPI_CHAR, rank,MPI_FREE_DATA, MPI_COMM_WORLD, &request);
}

void MpiServer::Discard(MPI_Status status){
	int count;
	std::unique_lock<std::mutex> lock(mpi_mutex_);
	MPI_Get_count(&status, MPI_BYTE, &count);
	char dummy[count];
	MPI_Recv(&dummy, count, MPI_CHAR, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	DLOG(FATAL) << "[" << rank_ << "] Discarding message " << status.MPI_TAG << ".\n";
}

void MpiServer::MPI_Terminate(){
	listen_ = false;
	int n = GetMpiNodeCount();
	//TODO(jlovitt): wait for mainloop to terminate
	std::unique_lock<std::mutex> lock(mpi_mutex_);
	for(int i = 1; i < n; i++ ){
		MPI_Send(NULL,0, MPI_BYTE, i, MPI_TERMINATE, MPI_COMM_WORLD);
	}
	MPI_Finalize();
}

} // end namespace minerva
#endif


