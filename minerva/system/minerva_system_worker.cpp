#include "minerva_system.h"
#include <cstdlib>
#include <mutex>
#include <cstring>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif
#include <dmlc/logging.h>
#include <gflags/gflags.h>
#include "backend/dag/dag_scheduler.h"
#include "backend/simple_backend.h"
#include "common/cuda_utils.h"

DEFINE_bool(use_dag, true, "Use dag engine");
DEFINE_bool(no_init_glog, false, "Skip initializing Google Logging");

using namespace std;

namespace minerva {

//TODO: 1 multiway memcopy!
void MinervaSystemWorker::UniversalMemcpy(
    pair<Device::MemType, uint64_t> to,
    pair<Device::MemType, uint64_t> from,
    size_t size) {
#ifdef HAS_CUDA
  CUDA_CALL(cudaMemcpy(to.second, from.second, size, cudaMemcpyDefault));
#else
  CHECK_EQ(static_cast<int>(to.first), static_cast<int>(Device::MemType::kCpu));
  CHECK_EQ(static_cast<int>(from.first), static_cast<int>(Device::MemType::kCpu));
  memcpy(to.second, from.second, size);
#endif
}

int const MinervaSystem::has_cuda_ =
#ifdef HAS_CUDA
1
#else
0
#endif
;

MinervaSystemWorker::~MinervaSystemWorker() {
  //delete backend_;
  delete device_manager_;
  delete profiler_;
}

//TODO: 1 Translate symbolic mpi ptr ids to local data ptr.
pair<Device::MemType, float*> MinervaSystemWorker::GetPtr(uint64_t device_id, uint64_t data_id) {
  return device_manager_->GetDevice(device_id)->GetPtr(data_id);
}

uint64_t MinervaSystemWorker::CreateCpuDevice() {
  return MinervaSystem::Instance().device_manager().CreateCpuDevice();
}
uint64_t MinervaSystemWorker::CreateGpuDevice(int id) {
  return MinervaSystem::Instance().device_manager().CreateGpuDevice(id);
}

void MinervaSystemWorker::SetDevice(uint64_t id) {
  current_device_id_ = id;
}

int rank(){
	return mpi_.rank();
}

MinervaSystemWorker::MinervaSystemWorker()
  : current_device_id_(0) {

  profiler_ = new ExecutionProfiler();
  device_manager_ = new DeviceManager();
  mpi_ = new Mpi();
  //backend_ = new SimpleBackend(*device_manager_);
}


}  // end of namespace minerva

