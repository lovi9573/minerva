
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
	int rank();
	void FinalizeTask(uint64_t);
private:
	void Default_Handler(uint64_t id, char* buffer, size_t size, int rank, int tag) override;
	void Handle_Device_Count(uint64_t id, char* buffer, size_t size, int rank );
	void Handle_Create_Device(uint64_t id, char* buffer, size_t size, int rank);
	void Handle_Task(uint64_t id, char* buffer, size_t size, int rank);
	void Handle_Free_Data(uint64_t id, char* buffer, size_t size, int rank);
	void Print_Profiler_Results();

};




} // end namespace minerva
#endif // end HAS_MPI

#endif /* MPI_MPI_HANDLER_H_ */
