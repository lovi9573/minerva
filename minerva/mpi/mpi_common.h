/*
 * mpi_common.h
 *
 *  Created on: Jun 10, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_MPI_MPI_COMMON_H_
#define MINERVA_MPI_MPI_COMMON_H_

#include <dmlc/logging.h>
//#define MPI_LOGGING
//#define MPI_DATA_LOGGING
//#define MPI_TASK_LOGGING

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
	#define MPI_FREE_DATA 7
	#define MPI_PRINT_PROFILE 8
	#define MPI_TERMINATE 100

	#ifdef MPI_LOGGING
		#define MPILOG std::cout
		#define MPILOG_DATA std::cout
		#define MPILOG_TASK std::cout
	#else
		#define MPILOG  true ? (void) 0 : dmlc::LogMessageVoidify() & LOG(INFO)

		#ifdef MPI_DATA_LOGGING
			#define MPILOG_DATA std::cout
		#else
			#define MPILOG_DATA  true ? (void) 0 : dmlc::LogMessageVoidify() & LOG(INFO)
		#endif

		#ifdef MPI_TASK_LOGGING
			#define MPILOG_TASK std::cout
		#else
			#define MPILOG_TASK  true ? (void) 0 : dmlc::LogMessageVoidify() & LOG(INFO)
		#endif
	#endif

#endif
} // end namespace minerva

#endif /* MINERVA_MPI_MPI_COMMON_H_ */
