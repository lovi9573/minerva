#pragma once
#include <unordered_map>
#include "device/device.h"
#include "device/device_listener.h"
#include "common/common.h"
#include "../mpi/mpi_server.h"

namespace minerva {

class DeviceManager {
 public:
  DeviceManager(bool);
  ~DeviceManager();
  uint64_t CreateCpuDevice();
  uint64_t CreateGpuDevice(int gid);
  uint64_t CreateMpiDevice(int rank, int gid);
#ifdef HAS_MPI
  uint64_t CreateCpuDevice(uint64_t device_id);
  uint64_t CreateGpuDevice(int gid, uint64_t device_id);
#endif
  int GetGpuDeviceCount();
  int GetMpiNodeCount();
  int GetMpiDeviceCount(int rank);
  Device* GetDevice(uint64_t id);
  void FreeData(uint64_t id);
  void RegisterListener(DeviceListener* l) { listener_ = l; }

 private:
  uint64_t GenerateDeviceId();
  DeviceListener* listener_;
  std::unordered_map<uint64_t, Device*> device_storage_;
  bool worker;
  DISALLOW_COPY_AND_ASSIGN(DeviceManager);
};

}  // namespace minerva

