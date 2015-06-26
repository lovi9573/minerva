#include "device_manager.h"
#include <dmlc/logging.h>
#include "system/minerva_system.h"
#include "device/device.h"
#include "common/cuda_utils.h"
#include "common/common.h"
#ifdef HAS_CUDA
#include <cuda.h>
#endif

using namespace std;

namespace minerva {

DeviceManager::DeviceManager(bool worker) : worker(worker) {
#ifdef HAS_CUDA
  GetGpuDeviceCount();  // initialize driver, ensure correct `atexit` order
#endif
}

DeviceManager::~DeviceManager() {
  for (auto i : device_storage_) {
    delete i.second;
  }
}

uint64_t DeviceManager::CreateCpuDevice() {
  auto id = GenerateDeviceId();
#ifdef HAS_MPI
  Device* d = new CpuDevice(MinervaSystem::Instance().rank(), id, listener_);
#else
  Device* d = new CpuDevice(id, listener_);
#endif

  CHECK(device_storage_.emplace(id, d).second);
  return id;
}

uint64_t DeviceManager::CreateCpuDevice(uint64_t id) {
#ifdef HAS_MPI
  Device* d = new CpuDevice(MinervaSystem::Instance().rank(), id, listener_);
#else
  Device* d = new CpuDevice(id, listener_);
#endif
  CHECK(device_storage_.emplace(id, d).second);
  return id;
}

uint64_t DeviceManager::CreateGpuDevice(int gid) {
#ifdef HAS_CUDA
  auto id = GenerateDeviceId();
#ifdef HAS_MPI
  Device* d = new GpuDevice(MinervaSystem::Instance().rank(), id, listener_, gid);
#else
  Device* d = new GpuDevice(id, listener_, gid);
#endif
  CHECK(device_storage_.emplace(id, d).second);
  return id;
#else
  common::FatalError("please recompile with macro HAS_CUDA");
#endif
}

uint64_t DeviceManager::CreateGpuDevice(int gid, uint64_t id) {
#ifdef HAS_CUDA
#ifdef HAS_MPI
  Device* d = new GpuDevice(MinervaSystem::Instance().rank(), id, listener_, gid);
#else
  Device* d = new GpuDevice(id, listener_, gid);
#endif
  CHECK(device_storage_.emplace(id, d).second);
  return id;
#else
  common::FatalError("please recompile with macro HAS_CUDA");
#endif
}

uint64_t DeviceManager::CreateMpiDevice(int rank, int gid) {
#ifdef HAS_MPI
  auto id = GenerateDeviceId();
  Device* d = new MpiDevice(rank, id, listener_, gid);
  CHECK(device_storage_.emplace(id, d).second);
  return id;
#else
  common::FatalError("please recompile with macro HAS_MPI");
#endif
}


int DeviceManager::GetGpuDeviceCount() {
#ifdef HAS_CUDA
  int ret;
  CUDA_CALL(cudaGetDeviceCount(&ret));
  return ret;
#else
  common::FatalError("please recompile with macro HAS_CUDA");
#endif
}

int DeviceManager::GetMpiNodeCount() {
#ifdef HAS_MPI
  return MinervaSystem::Instance().mpi_server().GetMpiNodeCount();
#else
  common::FatalError("please recompile with macro HAS_MPI");
#endif
}

int DeviceManager::GetMpiDeviceCount(int rank) {
#ifdef HAS_MPI
  return MinervaSystem::Instance().mpi_server().GetMpiDeviceCount(rank);
#else
  common::FatalError("please recompile with macro HAS_MPI");
#endif
}


Device* DeviceManager::GetDevice(uint64_t id) {
  CHECK_EQ(device_storage_.count(id), 1) << "id: " << id << ", rank: " << MinervaSystem::Instance().rank();
  return device_storage_.at(id);
}

void DeviceManager::FreeData(uint64_t id) {
  for (auto i : device_storage_) {
    i.second->FreeDataIfExist(id);
  }
}

uint64_t DeviceManager::GenerateDeviceId() {
	if(worker){
		LOG(FATAL) << "Cannot generate device id's in a worker device manager";
	}
  static uint64_t index_counter = 0;
  return index_counter++;
}

}  // namespace minerva

