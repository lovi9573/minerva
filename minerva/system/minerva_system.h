#pragma once
#include <atomic>
#include <memory>


#include "common/singleton.h"
#include "dag/physical_dag.h"
#include "backend/backend.h"
#include "device/device_manager.h"
#include "device/device.h"
#include "profiler/execution_profiler.h"
#include "system/minerva_system_interface.h"

#ifdef HAS_MPI
#include "mpi/mpi_handler.h"
#include "mpi/mpi_server.h"
#endif


namespace minerva {


class MinervaSystem :
  public IMinervaSystem,
  public common::EverlastingSingleton<MinervaSystem> {
  friend class common::EverlastingSingleton<MinervaSystem>;

 public:
  static void UniversalMemcpy(std::pair<Device::MemType, element_t*>, std::pair<Device::MemType, element_t*>, size_t);
  static void UniversalMemcpy(void* ,void*, size_t);
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


  void Request_Data(char* buffer, size_t bytes, int rank, uint64_t device_id, uint64_t data_id);
  std::pair<Device::MemType, element_t*> GetPtr(uint64_t, uint64_t);
  uint64_t GenerateDataId();
  uint64_t GenerateTaskId();

  // device
  uint64_t CreateCpuDevice();
  uint64_t CreateGpuDevice(int);
  uint64_t CreateFpgaDevice(int);
  uint64_t CreateMpiDevice(int, int);


  void SetDevice(uint64_t );
  uint64_t current_device_id() const { return current_device_id_; }
  // system
  void WaitForAll();
  int rank();
  void WorkerRun() override;

#ifdef HAS_MPI
  MpiServer& mpi_server(){
	  return *mpiserver_;
  }
  MpiHandler& mpi_handler(){
	  return *mpihandler_;
  }
  // device master
  void FreeMpiDataIfExist(int rank, uint64_t data_id);
  void FreeDataIfExist(uint64_t data_id);
#endif

#if defined(FIXED_POINT) || defined(HAS_FPGA)
  int get_fraction_width(){
	  return fraction_width;
  }
  void set_fraction_width(int w){
	  fraction_width = w;
  }
#endif




 private:
  MinervaSystem(int*, char***);
  PhysicalDag* physical_dag_;
  Backend* backend_;
  ExecutionProfiler* profiler_;
  DeviceManager* device_manager_;
  std::atomic<uint64_t> data_id_counter_;
  std::atomic<uint64_t> task_id_counter_;
  uint64_t current_device_id_;
  int rank_;
  bool worker_;
#ifdef HAS_MPI
  MpiHandler* mpihandler_;
  MpiServer* mpiserver_;
#endif
#if defined(FIXED_POINT) || defined(HAS_FPGA)
  int fraction_width;
#endif

};



}  // end of namespace minerva

