/*
 * mpi_common.h
 *
 *  Created on: Jun 10, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_MPI_MPI_COMMON_H_
#define MINERVA_MPI_MPI_COMMON_H_


namespace minerva {

#ifdef HAS_MPI


// MPI TAGS
#define MPI_DEVICE_COUNT 0
#define MPI_CREATE_DEVICE 1
#define MPI_TASK 2
#define MPI_TASK_DATA 3
#define MPI_TASK_DATA_REQUEST 4
#define MPI_TERMINATE 5


struct MpiTaskData{
	uint64_t task_id;
	uint64_t data_id;
	int owner_rank;
	int device_id;
	int dim;
};


struct MpiTask{
	uint64_t task_id;
	int worker_rank;
	int compute_device_id;
	int closure_type;
};

struct MpiDevices{
	MpiDevices(int r, int n): rank(r), nDevices(n){};
	int rank;
	int nDevices;
};


#endif



} // end namespace minerva

#endif /* MINERVA_MPI_MPI_COMMON_H_ */
