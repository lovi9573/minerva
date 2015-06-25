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
#define MPI_TASK_DATA_RESPONSE 5
#define MPI_FINALIZE_TASK 6
#define MPI_TERMINATE 7



#endif



} // end namespace minerva

#endif /* MINERVA_MPI_MPI_COMMON_H_ */
