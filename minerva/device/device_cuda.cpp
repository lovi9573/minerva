#include <utility>
#include <cstdlib>
#include <array>
#include <mutex>
#include <sstream>
#include <cstring>

#include <dmlc/logging.h>
#include <gflags/gflags.h>

#include "device.h"
#include "device/device_cuda.h"
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


using namespace std;

namespace minerva {


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
#endif // HAS_CUDA

}  // namespace minerva

