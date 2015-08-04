
#ifndef MPI_MPI_SERVER_H_
#define MPI_MPI_SERVER_H_


#include <cstring>
#include <cstddef>
#include "op/closure.h"
#include "device/task.h"
#include "op/context.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_data_handler.h"
#include "common/concurrent_unordered_set.h"

namespace minerva {

#ifdef HAS_MPI



class MpiServer: public MpiDataHandler {
public:
	void init();
	int rank();
	void Wait_On_Task(uint64_t);
	void MainLoop();
	void Handle_Finalize_Task(MPI_Status );
	int GetMpiNodeCount();
	int GetMpiDeviceCount(int rank);
	void CreateMpiDevice(int rank, int id, uint64_t);
	void MPI_Send_task(const Task& task,const Context& ctx );
	void MPI_Terminate();
private:
	int _rank;
	ConcurrentUnorderedSet<uint64_t> pending_tasks_;
	std::mutex task_complete_mutex_;
	std::condition_variable task_complete_condition_;
	bool listen_ = true;
};


#endif



} // end namespace minerva

#endif /* MPI_MPI_SERVER_H_ */
