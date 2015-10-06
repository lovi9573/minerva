

#include <cstring>
#include <cstddef>
#include <thread>
#include "op/closure.h"
#include "device/task.h"
#include "op/context.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_handler.h"
#include "system/minerva_system.h"
#include <unistd.h>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>


namespace minerva {

#ifdef HAS_MPI



MpiHandler::MpiHandler(int rank) : MpiDataHandler(rank){

}

/*
 *  ======== Publicly callable ========
 */

int MpiHandler::rank(){
	return rank_ ;
}

void MpiHandler::FinalizeTask(uint64_t task_id){
	MPILOG  << "[" << rank_  << "]=> {TODO} Sending Finalization message for task #" << task_id << "\n";
	char* buffer = new char[sizeof(uint64_t)];
	int offset = 0;
	SERIALIZE(buffer, offset, task_id, uint64_t)
	Send(buffer, sizeof(uint64_t), 0, MPI_FINALIZE_TASK);
	delete[] buffer;
}

/*
 *  ========= Default_Handler =========
 */

void MpiHandler::Default_Handler(uint64_t id, char* buffer, size_t size, int rank, int tag){
	switch(tag){
		case MPI_DEVICE_COUNT:
			Handle_Device_Count(id, buffer, size, rank);
			break;
		case MPI_CREATE_DEVICE:
			Handle_Create_Device(id, buffer, size, rank);
			break;
		case MPI_TASK:
			Handle_Task(id, buffer, size, rank);
			break;
		case MPI_FREE_DATA:
			Handle_Free_Data(id, buffer, size, rank);
			break;
		case MPI_PRINT_PROFILE:
			Print_Profiler_Results();
			break;
	}
}

/*
 *  ======== Receive Handlers =========
 */

void MpiHandler::Handle_Device_Count(uint64_t mpi_id, char* dummy, size_t size, int rank){
	int count = MinervaSystem::Instance().device_manager().GetGpuDeviceCount();
	char* buffer = new char[sizeof(uint64_t)];
	int offset = 0;
	SERIALIZE(buffer, offset, count, int)
	Send(mpi_id, buffer, sizeof(int), rank, MPI_DEVICE_COUNT);
	delete[] buffer;
}

void MpiHandler::Handle_Create_Device(uint64_t id, char* buffer, size_t size, int rank){
	int device_number;
	uint64_t device_id;
	int offset = 0;
	DESERIALIZE(buffer, offset, device_id, uint64_t)
	DESERIALIZE(buffer, offset, device_number, int)
	while (!MinervaSystem::IsAlive()){
		MPILOG  << "[" << rank_ << "] waiting for MinervaInstance to come alive.\n" ;
	}
	MPILOG  << "[" << rank_  << "]<= {mainloop} Creating device # "<< device_number << " device id: " << device_id << "\n";
	if(device_number == 0){
		MinervaSystem::Instance().device_manager().CreateCpuDevice(device_id);
	}else{
		MinervaSystem::Instance().device_manager().CreateGpuDevice(device_number-1, device_id);
	}
}

void MpiHandler::Handle_Task(uint64_t id, char* buffer, size_t size, int rank){
	int bytesconsumed = 0;
	Task& td = Task::DeSerialize(buffer,&bytesconsumed);
	MPILOG  << "[" << rank_  << "]<= {mainloop} Handling task #" << td.id << " for device id " << td.op.device_id << "\n";
	MinervaSystem::Instance().device_manager().GetDevice(td.op.device_id)->PushTask(&td);
}

void MpiHandler::Handle_Free_Data(uint64_t id, char* buffer, size_t size, int rank){
	uint64_t data_id;
	int offset = 0;
	DESERIALIZE(buffer, offset, data_id, uint64_t)
	MPILOG  << "[" << rank_  << "]<= {mainloop} Freeing data #" << data_id << "\n";
	MinervaSystem::Instance().FreeDataIfExist(data_id);
}

void MpiHandler::Print_Profiler_Results(){
	printf("RANK %d:\n",rank_);
	MinervaSystem::Instance().profiler().PrintResult();
}


#endif



} // end namespace minerva

