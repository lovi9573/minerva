/*
 * serialize.h
 *
 *  Created on: Jun 4, 2015
 *      Author: jlovitt
 */

#include <mpi.h>
#include "mpi_server.h"

#ifdef HAS_MPI

using namespace std;

namespace minerva {


MPI_Datatype MPI_TASKDATA;

void MpiServer::init(){
//	MPI_Init(0,NULL);
//	_rank = ::MPI::COMM_WORLD.Get_rank();

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
	*((int*)buffer) = id;
	*((uint64_t*)(buffer+sizeof(int))) = device_id;
	::MPI::COMM_WORLD.Send(buffer,size,::MPI::BYTE,rank,MPI_CREATE_DEVICE);
}


void MpiServer::MPI_Send_task(const Task& task,const Context& ctx ){
	DLOG(INFO) << "Task #"<< task.id << " being sent to rank #" << ctx.rank;
	size_t bufsize = task.GetSerializedSize();
	char buffer[bufsize];
	size_t usedbytes = task.Serialize(buffer);
	CHECK_EQ(bufsize, usedbytes);
	printf("++++mpisend");
	::MPI::COMM_WORLD.Send(buffer, bufsize, MPI_BYTE, ctx.rank, MPI_TASK);
	printf("++++mpisend-done");
}


#endif
}


