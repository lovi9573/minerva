
#ifndef MPI_MPI_HANDLER_H_
#define MPI_MPI_HANDLER_H_
#ifdef HAS_MPI

#include <mpi.h>
#include <cstring>
#include <cstddef>
#include <unordered_map>
#include "op/closure.h"
#include "device/task.h"
#include "op/context.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_data_handler.h"


namespace minerva {



class MpiHandler: public MpiDataHandler{
public:
	MpiHandler(int rank);
	void MainLoop();
	int rank();
	void FinalizeTask(uint64_t);
private:
	void Handle_Device_Count(::MPI::Status& );
	void Handle_Create_Device(::MPI::Status&);
	void Handle_Task(::MPI::Status&);
	//template<typename T> void Handle_Closure(MpiTask*, T* );

	//void PushReadyTasks();

	int _rank;
};




} // end namespace minerva
#endif // end HAS_MPI

#endif /* MPI_MPI_HANDLER_H_ */
