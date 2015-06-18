

#include <cstring>
#include <cstddef>
#include "op/closure.h"
#include "device/task.h"
#include "op/context.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_handler.h"
#include "system/minerva_system.h"



namespace minerva {

#ifdef HAS_MPI

#define CLOSURE_DONE 0x01
#define INPUTS_META_DONE 0x02
#define INPUTS_DIMS_DONE 0x04
#define OUTPUTS_META_DONE 0x08
#define OUTPUTS_DIMS_DONE 0x10
#define READY 0x1F


extern MPI_Datatype MPI_TASKDATA;

MpiHandler::MpiHandler(){
//	MPI_Init(0,NULL);
//	_rank = ::MPI::COMM_WORLD.Get_rank();
}

void MpiHandler::MainLoop(){
	bool term = false;
	::MPI::Status status;
	while (!term){
		::MPI::COMM_WORLD.Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, status);
		switch(status.Get_tag()){
		case MPI_DEVICE_COUNT:
			Handle_Device_Count(status);
			break;
		case MPI_CREATE_DEVICE:
			Handle_Create_Device(status);
			break;
		case MPI_TASK:
			Handle_Task(status);
			break;
		case MPI_TASK_DATA:
			Handle_Task_Data(status);
			break;
		case MPI_TASK_DATA_REQUEST:
			Handle_Task_Data_Request(status);
			break;
		case MPI_TERMINATE:
			term = true;
			break;
		}
		PushReadyTasks();
	}
}

int MpiHandler::rank(){
	return _rank;
}

void MpiHandler::PushReadyTasks(){
/*	//task_storage.LockRead();
	auto it = task_storage.begin();
	while (it != task_storage.end()){
		if ((it->second->readyness & READY) == READY){
			uint64_t device_id = it->second->task.op.device_id;
			MinervaSystemWorker::Instance().device_manager().GetDevice(device_id)->PushTask(&(it->second->task));
			it = task_storage.erase(it);
		}else{
			it++;
		}
	}
	//task_storage.UnLockRead();
	 *
	 */
}

void MpiHandler::Handle_Device_Count(::MPI::Status& status){
	int dummy;
	::MPI::COMM_WORLD.Recv(&dummy, 0, ::MPI::INT, status.Get_source(), MPI_DEVICE_COUNT);
	int count = MinervaSystem::Instance().device_manager().GetGpuDeviceCount();
	::MPI::COMM_WORLD.Send(&count, 1, MPI_INT, status.Get_source(), MPI_DEVICE_COUNT);
}

void MpiHandler::Handle_Create_Device(::MPI::Status& status){
	int id;
	uint64_t device_id;
	int count = status.Get_count(::MPI::BYTE);
	char buffer[count];
	::MPI::COMM_WORLD.Recv(buffer, count, ::MPI::BYTE, status.Get_source(), MPI_CREATE_DEVICE);
	id = *((int*)buffer);
	device_id = *((uint64_t*)(buffer+sizeof(int)));
	if(id == 0){
		MinervaSystem::Instance().device_manager().CreateCpuDevice(device_id);
	}else{
		MinervaSystem::Instance().device_manager().CreateGpuDevice(id-1, device_id);
	}
}

void MpiHandler::Handle_Task(::MPI::Status& status){
	int count = status.Get_count(MPI_BYTE);
	char bytes[count];
	::MPI::COMM_WORLD.Recv(&bytes, count, MPI_BYTE, status.Get_source(),MPI_TASK);
	//Task& td = Task::DeSerialize(bytes,0);

}


void MpiHandler::Handle_Task_Data(::MPI::Status& status){
	int count = status.Get_count(MPI_TASKDATA);
	MpiTaskData taskdata[count];
	::MPI::COMM_WORLD.Recv(&taskdata, count, MPI_TASKDATA, status.Get_source(),MPI_TASK_DATA );
	//TODO: 4 Put this task data somewhere...
}

void MpiHandler::Handle_Task_Data_Request(::MPI::Status& status){
	int count = status.Get_count(MPI_BYTE);
	char buffer[count];
	::MPI::COMM_WORLD.Recv(buffer, count, MPI_BYTE, status.Get_source(),MPI_TASK_DATA_REQUEST );
	//TODO: Fetch the task data and send.
}



#endif



} // end namespace minerva

