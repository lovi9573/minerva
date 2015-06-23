
#ifndef MPI_MPI_HANDLER_H_
#define MPI_MPI_HANDLER_H_

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

#ifdef HAS_MPI


class MpiHandler: public MpiDataHandler{
public:
	MpiHandler(int rank);
	void MainLoop();
	int rank();
private:
	void Handle_Device_Count(::MPI::Status& );
	void Handle_Create_Device(::MPI::Status&);
	void Handle_Task(::MPI::Status&);
	//template<typename T> void Handle_Closure(MpiTask*, T* );

	//void PushReadyTasks();

	int _rank;
};

#endif



} // end namespace minerva

#endif /* MPI_MPI_HANDLER_H_ */
