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

#define DEFAULT_POOL_SIZE ((size_t) 5.8 * 1024 * 1024 * 1024)
DEFINE_bool(no_execute, false, "Disable the actual computation (for performance debuggin)");

using namespace std;

namespace minerva {

Device::Device(uint64_t device_id, DeviceListener* l) : device_id_(device_id), data_store_{unique_ptr<DataStore>(nullptr)}, listener_(l) {
}

#ifdef HAS_MPI
Device::Device(int rank, uint64_t device_id, DeviceListener* l) : device_id_(device_id), _rank(rank), data_store_{unique_ptr<DataStore>(nullptr)}, listener_(l) {
}

int Device::rank(){
	return _rank;
}
#endif

pair<Device::MemType, float*> Device::GetPtr(uint64_t data_id) {
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
}

void ThreadedDevice::Execute(Task* task, int thrid) {
  PreExecute();
  //TODO: defer the ptr collection in the case of MPI call.
  if (GetMemType() == MemType::kMpi){
#ifdef HAS_MPI
	  auto& op = task->op;
	  CHECK(op.compute_fn);
	  if(!FLAGS_no_execute) {
		#ifndef NDEBUG
			Barrier(thrid);
			WallTimer calculate_timer;
			calculate_timer.Start();
		#endif
			DLOG(INFO) << Name() << " execute mpi task #" << task->id << ": " << op.compute_fn->Name();
			//TODO: 0 task has unique data_id's in it BUT how will workers be able to get the data?
			/*TODO: 1 The data local to this MPI node should be sent here instead of just the task.
			 *The message will be a combination of Task for peer data and DataShards for local data.
			 *This might a moot point for this embodiment of Minerva because I don't believe that any ops require more than one shard.
			 */
			DoExecute(task, thrid);
			DLOG(INFO) << Name() << " finished execute mpi task #" << task->id << ": " << op.compute_fn->Name();
		#ifndef NDEBUG
			calculate_timer.Stop();
			MinervaSystem::Instance().profiler().RecordTime(TimerType::kCalculation, op.compute_fn->Name(), calculate_timer);
		#endif
	  }
#else
	  common::FatalError("please recompile with macro HAS_MPI");
#endif
  }else{
	#ifndef NDEBUG
	  WallTimer memory_timer;
	  memory_timer.Start();
	#endif
	  /*
	   * Gather input data pointers
	   */
	  DataList input_shards;
	  for (auto& i : task->inputs) {
		auto& input_data = i.physical_data;
		if (input_data.device_id == device_id_) {  // Input is local
		  DLOG(INFO) << Name() << " input task data #" << i.id << " is local";
		  CHECK_EQ(local_data_.Count(input_data.data_id), 1);
		} else {
		  lock_guard<mutex> lck(copy_locks_[input_data.data_id]);
		  if (!remote_data_.Count(input_data.data_id)) {  // Input is remote and not copied
			DLOG(INFO) << Name() << " input task data #" << i.id << " is remote and not copied";
			size_t size = input_data.size.Prod() * sizeof(float);
			auto ptr = data_store_->CreateData(input_data.data_id, size);
			DoCopyRemoteData(ptr, MinervaSystem::Instance().GetPtr(input_data.device_id, input_data.data_id).second, size, thrid);
			CHECK(remote_data_.Insert(input_data.data_id));
		  }
		}
		//A list of (pointer , Scale) tuples.
		input_shards.emplace_back(data_store_->GetData(input_data.data_id), input_data.size);
	  }
	  /*
	   * Create output data pointers
	   */
	  DataList output_shards;
	  for (auto& i : task->outputs) {
		size_t size = i.physical_data.size.Prod() * sizeof(float);
		DLOG(INFO) << Name() << " create output for task data #" << i.id;
		auto ptr = data_store_->CreateData(i.physical_data.data_id, size);
		CHECK(local_data_.Insert(i.physical_data.data_id));
		output_shards.emplace_back(ptr, i.physical_data.size);
	  }
	  auto& op = task->op;
	  CHECK(op.compute_fn);
	  if(!FLAGS_no_execute) {
	#ifndef NDEBUG
		Barrier(thrid);
		memory_timer.Stop();
		MinervaSystem::Instance().profiler().RecordTime(TimerType::kMemory, op.compute_fn->Name(), memory_timer);
		WallTimer calculate_timer;
		calculate_timer.Start();
	#endif
		DLOG(INFO) << Name() << " execute task #" << task->id << ": " << op.compute_fn->Name();
		DoExecute(input_shards, output_shards, op, thrid);
		DLOG(INFO) << Name() << " finished execute task #" << task->id << ": " << op.compute_fn->Name();
	#ifndef NDEBUG
		calculate_timer.Stop();
		MinervaSystem::Instance().profiler().RecordTime(TimerType::kCalculation, op.compute_fn->Name(), calculate_timer);
	#endif
	  }
  }//end kCpu || kGpu
  listener_->OnOperationComplete(task);
} // end Execute

void ThreadedDevice::PreExecute() {
}

void ThreadedDevice::Barrier(int) {
}

#ifdef HAS_CUDA

struct GpuDevice::Impl {
  Impl(int);
  DISALLOW_COPY_AND_ASSIGN(Impl);
  ~Impl();
  inline void ActivateDevice() const;

  static size_t constexpr kParallelism = 4;
  int const device;
  array<cudaStream_t, kParallelism> stream;
  array<cublasHandle_t, kParallelism> cublas_handle;
  array<cudnnHandle_t, kParallelism> cudnn_handle;
};

GpuDevice::Impl::Impl(int d) : device(d) {
  ActivateDevice();
  for (size_t i = 0; i < kParallelism; ++i) {
    CUDA_CALL(cudaStreamCreate(&stream[i]));
    CUBLAS_CALL(cublasCreate(&cublas_handle[i]));
    CUBLAS_CALL(cublasSetStream(cublas_handle[i], stream[i]));
    CUDNN_CALL(cudnnCreate(&cudnn_handle[i]));
    CUDNN_CALL(cudnnSetStream(cudnn_handle[i], stream[i]));
  }
}

GpuDevice::Impl::~Impl() {
  ActivateDevice();
  for (size_t i = 0; i < kParallelism; ++i) {
    CUDNN_CALL(cudnnDestroy(cudnn_handle[i]));
    CUBLAS_CALL(cublasDestroy(cublas_handle[i]));
    CUDA_CALL(cudaStreamDestroy(stream[i]));
  }
}

void GpuDevice::Impl::ActivateDevice() const {
  CUDA_CALL(cudaSetDevice(device));
}

GpuDevice::GpuDevice(uint64_t device_id, DeviceListener* l, int gpu_id) : ThreadedDevice{device_id, l, Impl::kParallelism}, impl_{common::MakeUnique<Impl>(gpu_id)} {
  impl_->ActivateDevice();
  cudaFree(0);  // Initialize
  auto allocator = [this](size_t len) -> void* {
    void* ret;
    impl_->ActivateDevice();
    CUDA_CALL(cudaMalloc(&ret, len));
    return ret;
  };
  auto deallocator = [this](void* ptr) {
    impl_->ActivateDevice();
    CUDA_CALL(cudaFree(ptr));
  };
  data_store_ = common::MakeUnique<PooledDataStore>(DEFAULT_POOL_SIZE, allocator, deallocator);
}

GpuDevice::~GpuDevice() {
  impl_->ActivateDevice();
  pool_.WaitForAllFinished();
  // `data_store_` has to be deallocated before `impl_` does, because the `deallocator` of `data_store_` depends on `impl_`
  data_store_.reset();
}

Device::MemType GpuDevice::GetMemType() const {
  return MemType::kGpu;
}

string GpuDevice::Name() const {
  return common::FString("GPU device #%d", device_id_);
}

void GpuDevice::PreExecute() {
  impl_->ActivateDevice();
}

void GpuDevice::Barrier(int thrid) {
  CUDA_CALL(cudaStreamSynchronize(impl_->stream[thrid]));
}

void GpuDevice::DoCopyRemoteData(float* dst, float* src, size_t size, int thrid) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, impl_->stream[thrid]));
  CUDA_CALL(cudaStreamSynchronize(impl_->stream[thrid]));
}

void GpuDevice::DoExecute(const DataList& in, const DataList& out, PhysicalOp& op, int thrid) {
  Context ctx;
  ctx.impl_type = ImplType::kCuda;
  ctx.stream = impl_->stream[thrid];
  ctx.cublas_handle = impl_->cublas_handle[thrid];
  ctx.cudnn_handle = impl_->cudnn_handle[thrid];
  op.compute_fn->Execute(in, out, ctx);
  CUDA_CALL_MSG(op.compute_fn->Name(), cudaStreamSynchronize(impl_->stream[thrid]));
}

void GpuDevice::DoExecute(Task* task, int thrid){
	Execute(task, thrid);
}

#endif

#ifdef HAS_MPI
MpiDevice::MpiDevice(int rank, uint64_t device_id, DeviceListener* l, int gpu_id) : ThreadedDevice(device_id, l, kParallelism) , _gpu_id(gpu_id), _rank(rank){
	auto allocator = [](size_t len) -> void* {
	    void* ret = malloc(len);
	    return ret;
	  };
	  auto deallocator = [](void* ptr) {
	    free(ptr);
	  };
	  data_store_ = common::MakeUnique<DataStore>(allocator, deallocator);
}

MpiDevice::~MpiDevice(){
	pool_.WaitForAllFinished();
}

Device::MemType MpiDevice::GetMemType() const {
	return MemType::kMpi;
}

string MpiDevice::Name() const {
	return common::FString("Mpi Compute Node rank #%d",_rank);
}

/*
 * @param dst  The write pointer. Local to this device
 * @param src  The read pointer. Remote data.
 */
void MpiDevice::DoCopyRemoteData(float* dst, float* src, size_t size, int) {
	//TODO: 1 Copy data !
	// Determine data owner
	// Determine dst owner
	// Select copy method.
}

void MpiDevice::DoExecute(const DataList& in, const DataList& out, PhysicalOp& op, int thrid) {
	Context ctx;
	ctx.impl_type = ImplType::kMpi;
	ctx.rank = _rank;
	ctx.gpu_id = _gpu_id;
	//TODO: 1 Dispatch serialized op (code?), with datalists to _rank set to run on gpu _gpu_id;  Wait for response.
}

void MpiDevice::DoExecute(Task* task, int thrid){
	Context ctx;
	ctx.impl_type = ImplType::kMpi;
	ctx.rank = _rank;
	ctx.gpu_id = _gpu_id;

	task->op.compute_fn->Execute(*task, ctx);
}



#endif

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

void CpuDevice::DoCopyRemoteData(float* dst, float* src, size_t size, int) {
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
	Execute(task, thrid);
}

}  // namespace minerva

