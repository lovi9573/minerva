

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

#define CLOSURE_DONE 0x01
#define INPUTS_META_DONE 0x02
#define INPUTS_DIMS_DONE 0x04
#define OUTPUTS_META_DONE 0x08
#define OUTPUTS_DIMS_DONE 0x10
#define READY 0x1F


extern MPI_Datatype MPI_TASKDATA;

MpiHandler::MpiHandler(int rank) : MpiDataHandler(rank){
	//printf("MpiHandler initialized on rank %d\n",rank);
//	MPI_Init(0,NULL);
//	rank_  = ::MPI::COMM_WORLD.Getrank_ ();
}

void MpiHandler::MainLoop(){
	bool term = false;
	int pending_message = 0;
	MPI_Status status;
	//MPI_Status st;
	while (!term){
		{
			//TODO: Don't tie up all the mpi time doing IProbes.
			std::unique_lock<std::mutex> lock(mpi_mutex_);
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &pending_message, &status);
		}
		//MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
		if(pending_message){
			DLOG(INFO) << "[" << rank_  << "] Handling message" << status.MPI_TAG << ".\n";
			switch(status.MPI_TAG){
			case MPI_DEVICE_COUNT:
				DLOG(INFO) << "[" << rank_  << "] Fetching device count\n";
				Handle_Device_Count(status);
				break;
			case MPI_CREATE_DEVICE:
				Handle_Create_Device(status);
				break;
			case MPI_TASK:
				Handle_Task(status);
				break;
			case MPI_TASK_DATA_RESPONSE:
				Handle_Task_Data_Response(status);
				break;
			case MPI_TASK_DATA_REQUEST:
				Handle_Task_Data_Request(status);
				break;
			case MPI_FREE_DATA:
				Handle_Free_Data(status);
				break;
			case MPI_TERMINATE:
				term = true;
				break;
			}
			pending_message = 0;
		}
	}
	MPI_Finalize();
}

int MpiHandler::rank(){
	return rank_ ;
}


void MpiHandler::Handle_Device_Count(MPI_Status& status){
	int dummy;
	int count = MinervaSystem::Instance().device_manager().GetGpuDeviceCount();
	std::unique_lock<std::mutex> lock(mpi_mutex_);
	MPI_Recv(&dummy, 0, ::MPI::INT, status.MPI_SOURCE, MPI_DEVICE_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Send(&count, 1, MPI_INT, status.MPI_SOURCE, MPI_DEVICE_COUNT, MPI_COMM_WORLD);
}

void MpiHandler::Handle_Create_Device(MPI_Status& status){
	int id;
	uint64_t device_id;
	int count;
	std::unique_lock<std::mutex> lock(mpi_mutex_);
	MPI_Get_count(&status,MPI_BYTE, &count);
	char buffer[count];
	int offset = 0;
	MPI_Recv(buffer, count, ::MPI::BYTE, status.MPI_SOURCE, MPI_CREATE_DEVICE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	DESERIALIZE(buffer, offset, id, int)
	DESERIALIZE(buffer, offset, device_id, uint64_t)
	while (!MinervaSystem::IsAlive()){
		DLOG(INFO) << "[" << rank_ << "] waiting for MinervaInstance to come alive.\n" ;
	}
	DLOG(INFO) << "[" << rank_  << "] Creating device #" << id << "\n";
	if(id == 0){
		MinervaSystem::Instance().device_manager().CreateCpuDevice(device_id);
	}else{
		MinervaSystem::Instance().device_manager().CreateGpuDevice(id-1, device_id);
	}
}

void MpiHandler::Handle_Task(MPI_Status& status){
	std::unique_lock<std::mutex> lock(mpi_mutex_);
	int count;
	MPI_Get_count(&status,MPI_BYTE, &count);
	char bytes[count];
	MPI_Recv(&bytes, count, MPI_BYTE, status.MPI_SOURCE,MPI_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	int bytesconsumed = 0;
	Task& td = Task::DeSerialize(bytes,&bytesconsumed);
	DLOG(INFO) << "[" << rank_  << "] Handling task #" << td.id;
	MinervaSystem::Instance().device_manager().GetDevice(td.op.device_id)->PushTask(&td);
}


void MpiHandler::FinalizeTask(uint64_t task_id){
	std::unique_lock<std::mutex> lock(mpi_mutex_);
	DLOG(INFO) << "[" << rank_  << "] Sending Finalization message for task #" << task_id;
	MPI_Send(&task_id, sizeof(uint64_t), MPI_CHAR, 0, MPI_FINALIZE_TASK, MPI_COMM_WORLD);
}

void MpiHandler::Handle_Free_Data(MPI_Status& status){
	uint64_t data_id;
	int count = 0;
	MPI_Get_count(&status,MPI_BYTE, &count);
	MPI_Recv(&data_id, count, MPI_CHAR, status.MPI_SOURCE, MPI_FREE_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}


#endif



} // end namespace minerva

