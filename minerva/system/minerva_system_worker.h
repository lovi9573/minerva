#pragma once
#include <atomic>
#include <memory>
#include "common/singleton.h"
#include "dag/physical_dag.h"
#include "backend/backend.h"
#include "device/device_manager.h"
#include "device/device.h"
#include "profiler/execution_profiler.h"

namespace minerva {

class MinervaSystemWorker :
  public common::EverlastingSingleton<MinervaSystemWorker> {
  friend class common::EverlastingSingleton<MinervaSystemWorker>;

 public:
  static void UniversalMemcpy(std::pair<Device::MemType, float*>, std::pair<Device::MemType, float*>, size_t);
  static int const has_cuda_;
  MinervaSystemWorker() = delete;
  DISALLOW_COPY_AND_ASSIGN(MinervaSystemWorker);
  ~MinervaSystemWorker();

 /* Backend& backend() {
    return *backend_;
  }
  void wait_for_all()
  {
    backend_->WaitForAll();
  }
  */
  ExecutionProfiler& profiler() {
    return *profiler_;
  }
  DeviceManager& device_manager() {
    return *device_manager_;
  }
  std::pair<Device::MemType, float*> GetPtr(uint64_t, uint64_t);

  // device
  uint64_t CreateCpuDevice();
  uint64_t CreateGpuDevice(int);
  void SetDevice(uint64_t );
  uint64_t current_device_id() const { return current_device_id_; }
  // system
  void WaitForAll();
  int rank();

 private:
  MinervaSystemWorker();
  //Backend* backend_;
  ExecutionProfiler* profiler_;
  DeviceManager* device_manager_;
  Mpi* mpi_;
  std::unordered_map<uint64_t, Task> tasks;
  uint64_t current_device_id_;

};

}  // end of namespace minerva

