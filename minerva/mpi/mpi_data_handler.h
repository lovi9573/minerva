/*
 * MpiDataHandler.h
 *
 *  Created on: Jun 23, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_MPI_MPI_DATA_HANDLER_H_
#define MINERVA_MPI_MPI_DATA_HANDLER_H_

#ifdef HAS_MPI

#include <mpi.h>

namespace minerva {

class MpiDataHandler {
public:
	MpiDataHandler();
	virtual ~MpiDataHandler();
	void Request_Data(char*, size_t, int , uint64_t , uint64_t );
	void Handle_Task_Data(::MPI::Status&);
	void Handle_Task_Data_Request(::MPI::Status&);
};

} /* namespace minerva */

#endif

#endif /* MINERVA_MPI_MPI_DATA_HANDLER_H_ */
