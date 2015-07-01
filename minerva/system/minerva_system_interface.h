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

//TODO(jlovitt): Consider making a MinervaSystem interface to keep mpi bits hidden from owl.

class MinervaSystem :
  public common::EverlastingSingleton<MinervaSystem> {
  friend class common::EverlastingSingleton<MinervaSystem>;

 public:
  static void UniversalMemcpy(std::pair<Device::MemType, float*>, std::pair<Device::MemType, float*>, size_t);
  static void UniversalMemcpy(void* ,void*, size_t);
  static int const has_cuda_;
  static int const has_mpi_;
  static int const has_fpga_;
  MinervaSystem() = delete;
  DISALLOW_COPY_AND_ASSIGN(MinervaSystem);
  ~MinervaSystem();
  PhysicalDag& physical_dag();
  Backend& backend();
  void wait_for_all();
  ExecutionProfiler& profiler();
  DeviceManager& device_manager();


  void Request_Data(char* buffer, size_t bytes, int rank, uint64_t device_id, uint64_t data_id);
  std::pair<Device::MemType, float*> GetPtr(uint64_t, uint64_t);
  uint64_t GenerateDataId();
  uint64_t GenerateTaskId();

  // device
  uint64_t CreateCpuDevice();
  uint64_t CreateGpuDevice(int);
  uint64_t CreateFpgaDevice(int);
  uint64_t CreateMpiDevice(int, int);


  void SetDevice(uint64_t );
  uint64_t current_device_id() const ;
  // system
  void WaitForAll();
  int rank();
  void WorkerRun();
 private:
  MinervaSystem(int*, char***);

};



}  // end of namespace minerva

