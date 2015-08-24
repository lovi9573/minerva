#include <utility>
#include <cstdlib>
#include <array>
#include <mutex>
#include <sstream>
#include <cstring>

#include <dmlc/logging.h>
#include <gflags/gflags.h>

#include "device.h"
#include "device/device_mpi.h"
#include "device/device_fpga.h"
#include "device/device_cuda.h"
#include "device/task.h"
#include "system/minerva_system.h"
#include "op/context.h"
#include "common/cuda_utils.h"
#include "device/pooled_data_store.h"
#include "profiler/wall_timer.h"
#ifdef HAS_MPI
#include "mpi/mpi_handler.h"
#include "mpi/mpi_server.h"
#endif

using namespace std;

namespace minerva {


#ifdef HAS_MPI

/*
 *====================
 * 		BASE DEVICE
 * ====================
 */
Device::Device(int rank, uint64_t device_id, DeviceListener* l) : device_id_(device_id), data_store_{unique_ptr<DataStore>(nullptr)}, listener_(l), rank_(rank) {
}

int Device::rank(){
	return rank_;
}


/*
 *====================
 * 		MPI
 * ====================
 */
MpiDevice::MpiDevice(int rank, uint64_t device_id, DeviceListener* l, int gpu_id) : ThreadedDevice(device_id, l, kParallelism) , _gpu_id(gpu_id){
	auto allocator = [](size_t len) -> void* {
	    void* ret = malloc(len);
	    return ret;
	  };
	  auto deallocator = [](void* ptr) {
	    free(ptr);
	  };
	  data_store_ = common::MakeUnique<DataStore>(allocator, deallocator);
	  rank_ = rank;
}

MpiDevice::~MpiDevice(){
	pool_.WaitForAllFinished();
}

Device::MemType MpiDevice::GetMemType() const {
	return MemType::kMpi;
}

string MpiDevice::Name() const {
	return common::FString("Mpi Device shadowing Compute device %d at rank #%d", _gpu_id ,rank_);
}

/*
 * @param dst  The write pointer. Local to this device
 * @param src  The read pointer. Remote data.
 */
void MpiDevice::DoCopyRemoteData(element_t* dst, element_t* src, size_t size, int) {
	//TODO(jlovitt): Perhaps this will be possible with RDMA
	LOG(FATAL) << "Cannot copy over Mpi using pointers.";
}

void MpiDevice::Execute(Task* task, int thrid){
	PreExecute();
	auto& op = task->op;
	CHECK(op.compute_fn);

	#ifdef USE_PROFILER
		Barrier(thrid);
		WallTimer calculate_timer;
		calculate_timer.Start();
	#endif
	DLOG(INFO) << Name() << " dispatching mpi task #" << task->id << ": " << op.compute_fn->Name();
	//printf("Dispatching task < %lu > with new data size | %lu | to rank %d\n",task->id, task->outputs.size(), _rank);
	DoExecute(task, thrid);
	DLOG(INFO) << Name() << " notified of completion of mpi task #" << task->id << ": " << op.compute_fn->Name();
	#ifdef USE_PROFILER
		//printf("a\n");
		calculate_timer.Stop();
		//printf("b\n");
		MinervaSystem::Instance().profiler().RecordTime(TimerType::kMpi, op.compute_fn->Name(), calculate_timer);
	#endif
	//DLOG(INFO) << Name() << " notifying listener of completed task\n";
	listener_->OnOperationComplete(task);

}


void MpiDevice::DoExecute(const DataList& in, const DataList& out, PhysicalOp& op, int thrid) {
	LOG(FATAL) << "Cannot call DoExecute with DataList parameters on an MPI Shadow Device.";
}

void MpiDevice::DoExecute(Task* task, int thrid){
	Context ctx;
	ctx.impl_type = ImplType::kMpi;
	ctx.rank = rank_;
	ctx.gpu_id = _gpu_id;
	MinervaSystem::Instance().mpi_server().MPI_Send_task(*task, ctx );
	MinervaSystem::Instance().mpi_server().Wait_On_Task(task->id);
	//task->op.compute_fn->Execute(*task, ctx);
}

/*
 * ====================
 *   CPU
 * ====================
 */
CpuDevice::CpuDevice(int rank, uint64_t device_id, DeviceListener*l) : CpuDevice(device_id, l) {
	rank_ = rank;
}

#ifdef HAS_CUDA
/*
 *====================
 * 		GPU
 * ====================
 */
GpuDevice::GpuDevice(int rank, uint64_t device_id, DeviceListener* l, int gid) : GpuDevice(device_id, l, gid) {
	rank_ = rank;
}
#endif

#ifdef HAS_FPGA
/**
 * ====================
 * 	FPGA
 * ====================
 */
FpgaDevice::FpgaDevice(int rank, uint64_t device_id, DeviceListener* l, int sub_id) : FpgaDevice(device_id, l, sub_id) {
	rank_ = rank;
}
#endif

#endif //HAS_MPI
}  // namespace minerva

