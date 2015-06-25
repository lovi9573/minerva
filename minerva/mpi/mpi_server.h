
#ifndef MPI_MPI_SERVER_H_
#define MPI_MPI_SERVER_H_


#include <cstring>
#include <cstddef>
#include "op/closure.h"
#include "device/task.h"
#include "op/context.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_data_handler.h"

namespace minerva {

#ifdef HAS_MPI



class MpiServer: public MpiDataHandler {
public:
	void init();
	int rank();
	bool IsPending(uint64_t);
	void MainLoop();
	int GetMpiNodeCount();
	int GetMpiDeviceCount(int rank);
	void CreateMpiDevice(int rank, int id, uint64_t);
	void MPI_Send_task(const Task& task,const Context& ctx );
	void MPI_Send_task_data(const float* ptr, size_t size);
private:
	int _rank;
	ConcurrentUnorderedSet<uint64_t> _pendingTasks;
};


#endif



} // end namespace minerva

#endif /* MPI_MPI_SERVER_H_ */
