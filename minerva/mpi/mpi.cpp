/*
 * serialize.h
 *
 *  Created on: Jun 4, 2015
 *      Author: jlovitt
 */

#include "mpi/mpi.h"

namespace minerva {

#ifdef HAS_MPI
namespace mpi {


void MPI::MPI_init(){
	int size;

	MPI_Init();
	MPI_Comm_Size(MPI_COMM_WORLD,&size);


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

	MPI_Type_create_struct(2, blocklengths_data, offsets_data, types_data, &MpiTaskData);
	MPI_Type_commit(&mpi_car_type);

	// Create task Description struct
	MPI_Aint     offsets_desc[2];
	int blocklengths_desc[2] = {1,3};
#if __WORDSIZE == 64
    //typedef unsigned long int	uint64_t;
	MPI_Datatype types_desc[2] = {MPI_UNSIGNED_LONG, MPI_INT};
	offsets_desc[0] = offsetof(MpiTaskData, task_id);
	offsets_desc[1] = offsetof(MpiTaskData, worker_rank);
#else
	//TODO: 10 typedef unsigned long long int	uint64_t;

#endif
	MPI_Type_create_struct(2, blocklengths_desc, offsets_desc, types_desc, &MpiTask);
	MPI_Type_commit(&mpi_car_type);

}

void MPI_Send_task_descriptor(const Task& task, const Context& ctx, int closureType){
	struct mpi_task_descriptor td;
	td.task_id = task.id;
	td.worker_rank = ctx.rank;
	td.compute_device_id = ctx.gpu_id;
	td.closure_type = closureType;
	//TODO: 8 make async.
	mpi_send(&td, 1, mpi_task_descriptor_type, ctx.rank, MPI_TASK_DESCRIPTION, MPI_WORLD );
}

void MPI_Send_task_outputs(const Task& task,const Context& ctx ){
	int nOutputs = task.outputs.size();
	Mpi_Task_Data* dd = (Mpi_Task_Data*)malloc(sizeof(Mpi_Task_Data)*nInputs);
	int i;
	for(i=0; i<nIOutputs; i++){
		dd[i].dataid = task.outputs[i].id;
		dd[i].size = task.outputs[i].physical_data.size.Prod();
		//TODO: implement getrankfromdata_id
		dd[i].owner_rank = GetRankFromData_id(task.outputs[i].physical_data.data_id);
		dd[i].device_id = task.outputs[i].physical_data.device_id;
		dd[i].task_id = task.id;
	}
	mpi_send(&td, 1, mpi_task_data_type, ctx.rank, MPI_TASK_OUTPUTS, MPI_WORLD );
}


void MPI_Send_task_inputs(const Task& task,const Context& ctx ){
	int nInputs = task.inputs.size();
	Mpi_Task_Data* dd = (Mpi_Task_Data*)malloc(sizeof(Mpi_Task_Data)*nInputs);
	int i;
	for(i=0; i<nInputs; i++){
		dd[i].dataid = task.inputs[i].id;
		dd[i].size = task.inputs[i].physical_data.size.Prod();
		//TODO: implement getrankfromdata_id
		dd[i].owner_rank = GetRankFromData_id(task.inputs[i].physical_data.data_id);
		dd[i].device_id = task.inputs[i].physical_data.device_id;
		dd[i].task_id = task.id;
	}
	mpi_send(&td, 1, mpi_task_data_type, ctx.rank, MPI_TASK_INPUTS, MPI_WORLD );
}


}
#endif
}


