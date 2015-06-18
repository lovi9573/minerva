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


	// Create task Data struct
	MPI_Aint     offsets_data[2];
	int blocklengths_data[2] = {3,2};
#if __WORDSIZE == 64
    //typedef unsigned long int	uint64_t;
	MPI_Datatype types_data[2] = {MPI_UNSIGNED_LONG, MPI_INT};
	offsets_data[0] = offsetof(MpiTaskData, task_id);
	offsets_data[1] = offsetof(MpiTaskData, owner_rank);
#else
	//TODO: 10 typedef unsigned long long int	uint64_t;
#endif
	MPI_Type_create_struct(2, blocklengths_data, offsets_data, types_data, &MPI_TASKDATA);
	MPI_Type_commit(&MPI_TASKDATA);
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
	int count;
	::MPI::COMM_WORLD.Send(0,0,MPI_INT, rank,MPI_DEVICE_COUNT);
	::MPI::COMM_WORLD.Recv((void*)&count, 1, ::MPI::INT, rank,MPI_DEVICE_COUNT);
	return count;
}

void MpiServer::CreateMpiDevice(int rank, int id, uint64_t device_id){
	int size = sizeof(int)+sizeof(uint64_t);
	char buffer[size];
	*buffer = id;
	*(buffer+sizeof(int)) = device_id;
	::MPI::COMM_WORLD.Send(buffer,size,::MPI::BYTE,rank,MPI_CREATE_DEVICE);
}


void MpiServer::MPI_Send_task(const Task& task,const Context& ctx ){
	size_t bufsize = task.GetSerializedSize();
	char buffer[bufsize];
	bufsize = task.Serialize(buffer);
	::MPI::COMM_WORLD.Send(buffer, bufsize, MPI_BYTE, ctx.rank, MPI_TASK);
}


#endif
}


