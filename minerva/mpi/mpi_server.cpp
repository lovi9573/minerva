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

void MpiServer::init(){
//	MPI_Init(0,NULL);
//	_rank = ::MPI::COMM_WORLD.Get_rank();
	//_pendingTasks = ConcurrentUnorderedSet<uint64_t>();
}

void MpiServer::MainLoop(){
	bool term = false;
	::MPI::Status status;
	//MPI_Status st;
	while (!term){
		DLOG(INFO) << "[" << _rank << "] Top of mainloop.\n";
		::MPI::COMM_WORLD.Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, status);
		switch(status.Get_tag()){
		case MPI_TASK_DATA_REQUEST:
			Handle_Task_Data_Request(status);
			break;
		case MPI_FINALIZE_TASK:
			Handle_Finalize_Task(status);
			break;
		}
		//MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);

	}
}


int MpiServer::rank(){
	return _rank;
}

int MpiServer::GetMpiNodeCount(){
	return ::MPI::COMM_WORLD.Get_size();
}

int MpiServer::GetMpiDeviceCount(int rank){
	int size = ::MPI::COMM_WORLD.Get_size();
	if (rank >= size || rank == 0){
		return 0;
	}
	DLOG(INFO) << "Device count requested from rank #" << rank;
	int count;
	::MPI::COMM_WORLD.Send(0,0,MPI_INT, rank,MPI_DEVICE_COUNT);
	::MPI::COMM_WORLD.Recv((void*)&count, 1, ::MPI::INT, rank,MPI_DEVICE_COUNT);
	return count;
}

void MpiServer::CreateMpiDevice(int rank, int id, uint64_t device_id){
	DLOG(INFO) << "Creating device #" << id << " at rank "<< rank <<", uid " << device_id << "\n";
	int size = sizeof(int)+sizeof(uint64_t);
	char buffer[size];
	int offset = 0;
	SERIALIZE(buffer, offset, id, int)
	SERIALIZE(buffer, offset, device_id, uint64_t)
	::MPI::COMM_WORLD.Send(buffer,size,::MPI::BYTE,rank,MPI_CREATE_DEVICE);
}


void MpiServer::MPI_Send_task(const Task& task,const Context& ctx ){
	DLOG(INFO) << "Task #"<< task.id << " being sent to rank #" << ctx.rank;
	size_t bufsize = task.GetSerializedSize();
	char buffer[bufsize];
	size_t usedbytes = task.Serialize(buffer);
	CHECK_EQ(bufsize, usedbytes);
	::MPI::COMM_WORLD.Send(buffer, bufsize, MPI_BYTE, ctx.rank, MPI_TASK);
	_pending_tasks.Insert(task.id);
}


void MpiServer::Handle_Finalize_Task(::MPI::Status status){
	int count = status.Get_count(::MPI::BYTE);
	//char buffer[count];
	uint64_t task_id;
	::MPI::COMM_WORLD.Recv(&task_id, count, MPI_CHAR, status.Get_source(), MPI_FINALIZE_TASK);
	_pending_tasks.Erase(task_id);
	std::unique_lock<std::mutex> lock(_mutex);
	_task_complete_condition.notify_all();
}

void MpiServer::Wait_On_Task(uint64_t task_id){
	std::unique_lock<std::mutex> lock(_mutex);
	while(_pending_tasks.Count(task_id) > 0){
		_task_complete_condition.wait(lock);
	}
}


} // end namespace minerva
#endif


