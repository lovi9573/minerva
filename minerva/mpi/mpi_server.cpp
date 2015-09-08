/*
 * serialize.h
 *
 *  Created on: Jun 4, 2015
 *      Author: jlovitt
 */

#ifdef HAS_MPI


#include <mpi.h>
#include <thread>
#include "mpi_server.h"


using namespace std;

namespace minerva {



void MpiServer::printPendingTasks(){
	std::cout << "Pending Tasks:\n";
		for(auto i = pending_tasks_.begin(); i != pending_tasks_.end(); i++){
			std::cout << *i <<", ";
		}
	std::cout <<"\n";
}




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
	MPILOG << "Device count requested from rank #" << rank;
	int count;
	RecvItem r_item(reinterpret_cast<char*>(&count));
	uint64_t mpi_id = Get_Mpi_Id();
	{
		std::unique_lock<std::mutex> lock(recv_mutex_);
		recv_buffer_.insert( std::pair<uint64_t,RecvItem>(mpi_id, r_item) );
	}
	Send(mpi_id,0,0,rank,MPI_DEVICE_COUNT);
	Wait_For_Recv(mpi_id,reinterpret_cast<char*>(&count));
	return count;
}

void MpiServer::CreateMpiDevice(int rank, int id, uint64_t device_id){
	MPILOG << "Creating device #" << id << " at rank "<< rank <<", uid " << device_id << "\n";
	int size = sizeof(int)+sizeof(uint64_t);
	//TODO: delete this buffer when the message is sent.
	char* buffer = new char[size];
	int offset = 0;
	SERIALIZE(buffer, offset, device_id, uint64_t)
	SERIALIZE(buffer, offset, id, int)
	Send(buffer,size,rank,MPI_CREATE_DEVICE);
	delete[] buffer;
}


void MpiServer::MPI_Send_task(const Task& task,const Context& ctx ){
	MPILOG_TASK << "[0]=> {" << std::this_thread::get_id() << "} Task #"<< task.id << " being sent to rank #" << ctx.rank <<"\n";
	size_t bufsize = task.GetSerializedSize();
	char* buffer = new char[bufsize];
	size_t usedbytes = task.Serialize(buffer);
	CHECK_EQ(bufsize, usedbytes);
	{
		std::unique_lock<std::mutex> lock(task_complete_mutex_);
		pending_tasks_.insert(task.id);
//		printPendingTasks();
	}
	Send(buffer, bufsize, ctx.rank, MPI_TASK);
	//TODO: delete the buffer only after the message has been sent.
	delete[] buffer;
	//TODO:(jesselovitt) Maybe a wait_for_recv would do the trick here.
}

void MpiServer::Wait_On_Task(uint64_t task_id){
	MPILOG_TASK << "[0]== {" << std::this_thread::get_id() << "} waiting on task "<< task_id << "\n";
	std::unique_lock<std::mutex> lock(task_complete_mutex_);
//	printPendingTasks();
	while(pending_tasks_.count(task_id) > 0){
		task_complete_condition_.wait(lock);
	}
	MPILOG_TASK << "[0]== {" << std::this_thread::get_id() << "} awoken on completed task "<< task_id << "\n";
}

void MpiServer::Free_Data(int rank, uint64_t data_id){
	MPILOG << "[0]=> {" << std::this_thread::get_id() << "} Free data "<< data_id << "\n";
	char* buffer = new char[sizeof(uint64_t)];
	int offset = 0;
	SERIALIZE(buffer, offset, data_id, uint64_t)
	//TODO: delete the buffer only after the message has been sent.
	Send(buffer, offset, rank, MPI_FREE_DATA);
	delete[] buffer;
}

/*
 *  ========= Default_Handler =========
 */

void MpiServer::Default_Handler(uint64_t id, char* buffer, size_t size, int rank, int tag){
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
	int offset = 0;
	uint64_t task_id;
	DESERIALIZE(buffer, offset, task_id, uint64_t)
	MPILOG_TASK << "[0]<= {mainloop} Finalizing task "<< task_id << "\n";
	{
		std::unique_lock<std::mutex> task_lock(task_complete_mutex_);
		pending_tasks_.erase(task_id);
//		printPendingTasks();
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


