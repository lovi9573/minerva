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
#include <queue>

namespace minerva {

class SendItem {
public:
	SendItem(uint64_t i,char* b, size_t s, int d, int t):id(i),buffer(b),size(s),dest_rank(d),tag(t){};
	uint64_t id;
	char* buffer;
	size_t size;
	int dest_rank;
	int tag;
};

class RecvItem {
public:
	RecvItem(char* b):buffer(b), ready(0){};
	char* buffer;
	int ready;
};


class MpiDataHandler {
public:
	MpiDataHandler(int);
	virtual ~MpiDataHandler();
	void Request_Data(char*, size_t, int , uint64_t , uint64_t );
	void MainLoop();
	int rank();
protected:
	virtual void Default_Handler(uint64_t id, char* buffer, size_t size, int rank, int tag);
	void Handle_Task_Data_Response(uint64_t id, char* buffer, size_t size, int rank);
	void Handle_Task_Data_Request(uint64_t id, char* buffer, size_t size, int rank);
	uint64_t Send(char* msgbuffer, int size, int rank, int tag);
	void Wait_For_Recv(uint64_t mpi_id, char* buffer);
	int rank_;
	std::mutex id_mutex_;
	std::mutex send_mutex_;
	std::mutex recv_mutex_;
	std::condition_variable recv_complete_;
	std::map<uint64_t, RecvItem> recv_buffer_;
private:
	uint64_t Get_Mpi_Id();
	uint64_t id_;
	int id_stride_;
	MPI_Request send_request_;
	int send_complete_ ;
	std::queue<SendItem> send_queue_;
};

} /* namespace minerva */

#endif

#endif /* MINERVA_MPI_MPI_DATA_HANDLER_H_ */
