/*
 * serialize.h
 *
 *  Created on: Jun 4, 2015
 *      Author: jlovitt
 */

#ifndef MPI_MPI_H_
#define MPI_MPI_H_

namespace minerva {


#ifdef HAS_MPI
namespace mpi {

#define MPI_TASK_INPUT_DATA 0
#define MPI_TASK_OUTPUT_DATA 1
#define MPI_TASK_DESCRIPTION 2
#define MPI_TASK_CLOSURE 3

struct task_data_descriptor{
	uint64_t task_id;
	uint64_t dataid;
	uint64_t size;
	int owner_rank;
	int device_id;
};

//TODO: 1 Life will be much easier if the closure type and the closure data come together.
struct task_descriptor{
	uint64_t task_id;
	int worker_rank;
	int compute_device_id;
	int closure_type;
};

typedef struct task_data_descriptor MpiTaskData;
typedef struct task_descriptor MpiTask;


class MPI{
public:
	void MPI_init();
	void MPI_Send_task_descriptor(const Task& task, int closureType, const Context& ctx );
	void MPI_Send_task_inputs(const Task& task,const Context& ctx );
	void MPI_Send_task_outputs(const Task& task,const Context& ctx );
private:
	MPI_Datatype mpi_task_data_type;
	MPI_Datatype mpi_task_desc_type;
} //end MPI

} //end namespace mpi
#endif



}

#endif /* MPI_MPI_H_ */
