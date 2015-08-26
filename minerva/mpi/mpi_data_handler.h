/*
 * MpiDataHandler.h
 *
 *  Created on: Jun 23, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_MPI_MPI_DATA_HANDLER_H_
#define MINERVA_MPI_MPI_DATA_HANDLER_H_

#ifdef HAS_MPI
#include <mutex>
#include <mpi.h>
#include <condition_variable>

namespace minerva {

class MpiDataHandler {
public:
	MpiDataHandler(int);
	virtual ~MpiDataHandler();
	void Request_Data(char*, size_t, int , uint64_t , uint64_t );
protected:
	void Handle_Task_Data_Response(MPI_Status status);
	void Handle_Task_Data_Request(MPI_Status&);
	std::mutex mpi_mutex_;
	std::condition_variable mpi_receive_complete_;
	std::condition_variable mpi_request_complete_;
	int rank_;
private:
	char* pending_data_buffer;
	uint64_t pending_data_id;
	MPI_Request fulfillment_request;
	int fulfillment_complete ;
};

} /* namespace minerva */

#endif

#endif /* MINERVA_MPI_MPI_DATA_HANDLER_H_ */
