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

class MinervaSystem :
  public common::EverlastingSingleton<MinervaSystem> {
  friend class common::EverlastingSingleton<MinervaSystem>;

 public:
  static void UniversalMemcpy(std::pair<Device::MemType, float*>, std::pair<Device::MemType, float*>, size_t);
  static int const has_cuda_;
  static int const has_mpi_;
  MinervaSystem() = delete;
  DISALLOW_COPY_AND_ASSIGN(MinervaSystem);
  ~MinervaSystem();
  PhysicalDag& physical_dag() {
    return *physical_dag_;
  }
  Backend& backend() {
    return *backend_;
  }
  void wait_for_all()
  {
    backend_->WaitForAll();
  }
  ExecutionProfiler& profiler() {
    return *profiler_;
  }
  DeviceManager& device_manager() {
    return *device_manager_;
  }
  std::pair<Device::MemType, float*> GetPtr(uint64_t, uint64_t);
  uint64_t GenerateDataId();
  uint64_t GenerateTaskId();
#ifdef HAS_MPI
  // device
  uint64_t CreateCpuDevice(int);
  uint64_t CreateGpuDevice(int, int);
  uint64_t CreateMpiDevice(int, int, int);

#else
  // device
  uint64_t CreateCpuDevice();
  uint64_t CreateGpuDevice(int);
  uint64_t CreateMpiDevice(int, int);
#endif //end HAS_MPI
  void SetDevice(uint64_t );
  uint64_t current_device_id() const { return current_device_id_; }
  // system
  void WaitForAll();

 private:
  MinervaSystem(int*, char***);
  MinervaSystem(bool worker);
  PhysicalDag* physical_dag_;
  Backend* backend_;
  ExecutionProfiler* profiler_;
  DeviceManager* device_manager_;
  std::atomic<uint64_t> data_id_counter_;
  std::atomic<uint64_t> task_id_counter_;
  uint64_t current_device_id_;
#ifdef HAS_MPI
  bool _worker;
  bool _distributed;
#endif


};

}  // end of namespace minerva

