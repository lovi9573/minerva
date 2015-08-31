#include <utility>
#include <cstdlib>
#include <array>
#include <mutex>
#include <sstream>
#include <cstring>

#include <dmlc/logging.h>
#include <gflags/gflags.h>
#include "device.h"
#include "device/task.h"
#include "system/minerva_system.h"
#include "op/context.h"
#include "common/cuda_utils.h"
#include "device/pooled_data_store.h"
#include "profiler/wall_timer.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cuda.h>
#include <cublas_v2.h>
#endif
#ifdef HAS_MPI
#include "mpi/mpi_handler.h"
#include "mpi/mpi_server.h"
#endif

//#define DEFAULT_POOL_SIZE ((size_t) 5.8 * 1024 * 1024 * 1024)
DEFINE_bool(no_execute, false, "Disable the actual computation (for performance debuggin)");

using namespace std;

namespace minerva {

Device::Device(uint64_t device_id, DeviceListener* l) : device_id_(device_id), data_store_{unique_ptr<DataStore>(nullptr)}, listener_(l), rank_(0) {
}


pair<Device::MemType, element_t*> Device::GetPtr(uint64_t data_id) {
	//printf("[%d] Getting pointer to data id %lu\n",_rank, data_id);
  return make_pair(GetMemType(), data_store_->GetData(data_id));
}

void Device::FreeDataIfExist(uint64_t data_id) {
  auto d = local_data_.Erase(data_id) + remote_data_.Erase(data_id);
  if (d == 1) {
    data_store_->FreeData(data_id);
  } else if (d != 0) {
    LOG(FATAL) << "duplicate data";
  }
}

string Device::GetMemUsage() const {
  return common::FString("device #%d used %dB", device_id_, data_store_->GetTotalBytes());
}

ThreadedDevice::ThreadedDevice(uint64_t device_id, DeviceListener* l, size_t parallelism) : Device(device_id, l), pool_(parallelism) {
}

void ThreadedDevice::PushTask(Task* task) {
  if (!task->light)
    pool_.Push(bind(&ThreadedDevice::Execute, this, task, placeholders::_1));
  else
    // light weight tasks are executed directly to avoid thread switching
    Execute(task, 0);
}

void ThreadedDevice::FreeDataIfExist(uint64_t data_id) {
  copy_locks_.Erase(data_id);
  Device::FreeDataIfExist(data_id);
  //printf("[%d] Free Dev:%lu, DataSize:%lu\n",rank_, device_id_,data_store_->data_states_.size());
}

void ThreadedDevice::Execute(Task* task, int thrid) {
  PreExecute();
  if (GetMemType() == MemType::kMpi){
	  common::FatalError("Cannot call ThreadedDevice::Execute on MPI device.");
  }else{
	#ifdef USE_PROFILER
	  WallTimer memory_timer;
	  memory_timer.Start();
	#endif
	  /*
	   * Gather input data pointers
	   */
	  DataList input_shards;
	  for (auto& i : task->inputs) {
		auto& input_data = i.physical_data;
		//printf("[%d] process task input\n",_rank);
		if (input_data.device_id == device_id_) {  // Input is local
		  DLOG(INFO) << Name() << " input task data #" << input_data.data_id << " is local";
		  //printf("[%d] Task < %lu > Local data [ %lu ] count: %lu\n",_rank,task->id,input_data.data_id, local_data_.Count(input_data.data_id));
		  CHECK_EQ(local_data_.Count(input_data.data_id), 1); // << " rank: "<< _rank;
		} else {
		  lock_guard<mutex> lck(copy_locks_[input_data.data_id]);
		  if (!remote_data_.Count(input_data.data_id)) {  // Input is remote and not copied
			DLOG(INFO) << Name() << " input task data #" << input_data.data_id << " is remote and not copied";
			size_t size = input_data.size.Prod() * sizeof(element_t);
			auto ptr = data_store_->CreateData(input_data.data_id, size);
			//printf("[%d] Data ptr returned\n",rank_);
#ifdef HAS_MPI
			if(MinervaSystem::has_mpi_ == 1 && input_data.rank != MinervaSystem::Instance().rank()){
				//printf("[%d] Device requesting data for %lu element_ts\n",rank_, size);
				MinervaSystem::Instance().Request_Data(reinterpret_cast<char*>(ptr), size, input_data.rank,  input_data.device_id, input_data.data_id );
			}else{
				//printf("[%d] Copying Intra-rank data for %lu element_ts\n",rank_, size);
				DoCopyRemoteData(ptr, MinervaSystem::Instance().GetPtr(input_data.device_id, input_data.data_id).second, size, thrid);
			}
#else
			DoCopyRemoteData(ptr, MinervaSystem::Instance().GetPtr(input_data.device_id, input_data.data_id).second, size, thrid);
#endif
			CHECK(remote_data_.Insert(input_data.data_id));
		  }
		}
		//A list of (pointer , Scale) tuples.
		//printf("[%d] Placing input shard id %lu\n",rank_, input_data.data_id);
		input_shards.emplace_back(data_store_->GetData(input_data.data_id), input_data.size);
	  }
	  /*
	   * Create output data pointers
	   */
	  //printf("[%d] creating output shards for task < %lu > with | %lu | outputs\n",rank_, task->id, task->outputs.size());
	  DataList output_shards;
	  for (auto& i : task->outputs) {
		size_t size = i.physical_data.size.Prod() * sizeof(element_t);
		//printf("[%d] Device creating output shard for %lu element_ts\n",_rank, size);
		DLOG(INFO) << Name() << " create output for task data #" << i.physical_data.data_id;
		//printf("[%d] Local data [ %lu ] created on rank %d\n",_rank,i.physical_data.data_id, _rank);
		auto ptr = data_store_->CreateData(i.physical_data.data_id, size);
		CHECK(local_data_.Insert(i.physical_data.data_id));
		output_shards.emplace_back(ptr, i.physical_data.size);
	  }
	  auto& op = task->op;
	  CHECK(op.compute_fn);
	  if(!FLAGS_no_execute) {
	#ifdef USE_PROFILER
		Barrier(thrid);
		memory_timer.Stop();
		MinervaSystem::Instance().profiler().RecordTime(TimerType::kMemory, op.compute_fn->Name(), memory_timer);
		WallTimer calculate_timer;
		calculate_timer.Start();
	#endif
		DLOG(INFO) << Name() << " execute task #" << task->id << ": " << op.compute_fn->Name();
		DoExecute(input_shards, output_shards, op, thrid);
		DLOG(INFO) << Name() << " finished execute task #" << task->id << ": " << op.compute_fn->Name();
	#ifdef USE_PROFILER
		calculate_timer.Stop();
		MinervaSystem::Instance().profiler().RecordTime(TimerType::kCalculation, op.compute_fn->Name(), calculate_timer);
	#endif
	  }
  }//end kCpu || kGpu

#ifdef HAS_MPI
  if(MinervaSystem::has_mpi_ == 1 && MinervaSystem::Instance().rank() != 0){
	  MinervaSystem::Instance().mpi_handler().FinalizeTask(task->id);
  }else{
	  //DLOG(INFO) << Name() << " notifying listener of completed task\n";
	  listener_->OnOperationComplete(task);
  }
#else
  listener_->OnOperationComplete(task);
#endif

} // end Execute

void ThreadedDevice::PreExecute() {
}

void ThreadedDevice::Barrier(int) {
}




/*
 * ====================
 * 		CPU
 * ====================
 */
CpuDevice::CpuDevice(uint64_t device_id, DeviceListener* l) : ThreadedDevice(device_id, l, kDefaultThreadNum) {
  auto allocator = [](size_t len) -> void* {
    void* ret = malloc(len);
    return ret;
  };
  auto deallocator = [](void* ptr) {
    free(ptr);
  };
  data_store_ = common::MakeUnique<DataStore>(allocator, deallocator);
}

CpuDevice::~CpuDevice() {
  pool_.WaitForAllFinished();
}

Device::MemType CpuDevice::GetMemType() const {
  return MemType::kCpu;
}

string CpuDevice::Name() const {
  return common::FString("CPU device #%d", device_id_);
}

void CpuDevice::DoCopyRemoteData(element_t* dst, element_t* src, size_t size, int) {
#ifdef HAS_CUDA
  CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
#else
  memcpy(dst, src, size);
#endif
}

void CpuDevice::DoExecute(const DataList& in, const DataList& out, PhysicalOp& op, int) {
  Context ctx;
  ctx.impl_type = ImplType::kBasic;
  op.compute_fn->Execute(in, out, ctx);
}

void CpuDevice::DoExecute(Task* task, int thrid){
	LOG(FATAL) << "Cannot call DoExecute with Task* parameter on a CPU Device.";
}

}  // namespace minerva

