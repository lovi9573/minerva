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


/*
 *  ======== Publicly callable ========
 */
void MpiServer::init(){
}


int MpiServer::GetMpiNodeCount(){
	int size = 0;
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
	uint64_t id = Send(0,0,rank,MPI_DEVICE_COUNT);
	Wait_For_Recv(id,reinterpret_cast<char*>(&count));
	return count;
}

void MpiServer::CreateMpiDevice(int rank, int id, uint64_t device_id){
	DLOG(INFO) << "Creating device #" << id << " at rank "<< rank <<", uid " << device_id << "\n";
	int size = sizeof(int)+sizeof(uint64_t);
	//TODO: delete this buffer when the message is sent.
	char* buffer = new char[size];
	int offset = 0;
	SERIALIZE(buffer, offset, id, int)
	SERIALIZE(buffer, offset, device_id, uint64_t)
	Send(buffer,size,rank,MPI_CREATE_DEVICE);
}


void MpiServer::MPI_Send_task(const Task& task,const Context& ctx ){
	DLOG(INFO) << "Task #"<< task.id << " being sent to rank #" << ctx.rank;
	size_t bufsize = task.GetSerializedSize();
	char* buffer = new char[bufsize];
	size_t usedbytes = task.Serialize(buffer);
	CHECK_EQ(bufsize, usedbytes);
	Send(buffer, bufsize, ctx.rank, MPI_TASK);
	//TODO: delete the buffer only after the message has been sent.
	//delete[] buffer;
	//TODO:(jesselovitt) Maybe a wait_for_recv would do the trick here.
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

void MpiServer::Free_Data(int rank, uint64_t data_id){
	char* buffer = new char[sizeof(uint64_t)];
	int offset = 0;
	SERIALIZE(buffer, offset, data_id, uint64_t)
	//TODO: delete the buffer only after the message has been sent.
	Send(buffer, offset, rank, MPI_FREE_DATA);
}

/*
 *  ========= Default_Handler =========
 */

void MpiServer::Default_Handler(uint64_t id, char* buffer, size_t size, int rank, int tag){
	DLOG(INFO) << "[" << rank_ << "] Handling message " << tag ;
	switch(tag){
		case MPI_FINALIZE_TASK:
			Handle_Finalize_Task(id, buffer, size, rank);
			break;
		case MPI_DEVICE_COUNT:
			Handle_Device_Count(id, buffer, size, rank);
			break;
	}
}

/*
 *  ======== Receive Handlers =========
 */

void MpiServer::Handle_Finalize_Task(uint64_t id, char* buffer, size_t size, int rank ){
	uint64_t task_id = *(reinterpret_cast<uint64_t*>(buffer));
	{
		std::unique_lock<std::mutex> task_lock(task_complete_mutex_);
		pending_tasks_.Erase(task_id);
	}
	task_complete_condition_.notify_all();
}

void MpiServer::Handle_Device_Count(uint64_t id, char* buffer, size_t size, int rank){
	std::unique_lock<std::mutex> lock(recv_mutex_);
	if(recv_buffer_.find(id) != recv_buffer_.end()){
		recv_buffer_.at(id).ready = 1;
		std::memcpy(recv_buffer_.at(id).buffer, buffer, size);
	}
}



void MpiServer::MPI_Terminate(){
	listen_ = false;
	int n = GetMpiNodeCount();
	//TODO(jlovitt): wait for mainloop to terminate
	for(int i = 1; i < n; i++ ){
		Send(NULL,0, i, MPI_TERMINATE);
	}
	//TODO(JesseLovitt):Wait for send queue to empty.
	MPI_Finalize();
}

} // end namespace minerva
#endif


