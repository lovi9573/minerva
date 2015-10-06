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


class IMinervaSystem {
 public:
  static IMinervaSystem& Interface();
  static void Init(int*, char***);
  static int const has_cuda_;
  static int const has_mpi_;
  static int const has_fpga_;
  virtual PhysicalDag& physical_dag() = 0;
  virtual Backend& backend() = 0;
  virtual void wait_for_all() = 0;
  virtual ExecutionProfiler& profiler() = 0;
  virtual DeviceManager& device_manager() = 0;

  virtual std::pair<Device::MemType, element_t*> GetPtr(uint64_t, uint64_t) = 0;
  virtual uint64_t GenerateDataId() = 0;
  virtual uint64_t GenerateTaskId() = 0;

  // device
  virtual uint64_t CreateCpuDevice() = 0;
  virtual uint64_t CreateGpuDevice(int) = 0;
  virtual uint64_t CreateFpgaDevice(int) = 0;
  virtual uint64_t CreateMpiDevice(int, int) = 0;


  virtual void SetDevice(uint64_t ) = 0;
  virtual uint64_t current_device_id() const  = 0;
  // system
  virtual void WaitForAll() = 0;
  virtual int rank() = 0;
  virtual void WorkerRun() = 0;
  virtual void PrintProfilerResults() = 0;

};



}  // end of namespace minerva

