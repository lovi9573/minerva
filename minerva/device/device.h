#pragma once
#include <string>
#include <utility>
#include <mutex>
#include <memory>
#include "device/task.h"
#include "device/data_store.h"
#include "device/device_listener.h"
#include "op/physical_fn.h"
#include "common/common.h"
#include "common/thread_pool.h"
#include "common/concurrent_blocking_queue.h"
#include "common/concurrent_unordered_set.h"
#include "common/concurrent_unordered_map.h"

namespace minerva {

/*
 * Base class for a device
 */
class Device {
 public:
  enum class MemType {
    kCpu,
    kGpu,
    kMpi,
  };
  Device() = delete;
  DISALLOW_COPY_AND_ASSIGN(Device);
  Device(uint64_t device_id, DeviceListener*);
  virtual ~Device() = default;
  virtual void PushTask(Task*) = 0;
  virtual std::pair<MemType, float*> GetPtr(uint64_t data_id);
  virtual void FreeDataIfExist(uint64_t data_id);
  virtual std::string GetMemUsage() const;
  virtual std::string Name() const = 0;
  virtual MemType GetMemType() const = 0;
#ifdef HAS_MPI
  Device(int rank, uint64_t device_id, DeviceListener* l);
  int rank();
#endif

 protected:
  /*
   * Set of local data uids
   */
  ConcurrentUnorderedSet<uint64_t> local_data_;
  /*
   * Set of cached remote data uids
   */
  ConcurrentUnorderedSet<uint64_t> remote_data_;
  uint64_t device_id_;
#ifdef HAS_MPI
  int _rank;
#endif
  std::unique_ptr<DataStore> data_store_;
  DeviceListener* listener_;
};

class ThreadedDevice : public Device {
 public:
  ThreadedDevice() = delete;
  ThreadedDevice(uint64_t device_id, DeviceListener*, size_t parallelism);
#ifdef HAS_MPI
  ThreadedDevice(int rank, uint64_t device_id, DeviceListener*, size_t parallelism);
#endif
  DISALLOW_COPY_AND_ASSIGN(ThreadedDevice);
  ~ThreadedDevice() = default;
  void PushTask(Task*) override;
  void FreeDataIfExist(uint64_t data_id) override;

 protected:
  virtual void Execute(Task*, int thrid);
  virtual void PreExecute();
  virtual void Barrier(int);
  virtual void DoCopyRemoteData(float*, float*, size_t, int) = 0;
  virtual void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) = 0;
  ConcurrentUnorderedMap<uint64_t, std::mutex> copy_locks_;
  ThreadPool pool_;
};

#ifdef HAS_CUDA
class GpuDevice : public ThreadedDevice {
 public:
  GpuDevice(uint64_t device_id, DeviceListener*, int gpu_id);
#ifdef HAS_MPI
  GpuDevice(int rank, uint64_t device_id, DeviceListener*, int gpu_id);
#endif
  DISALLOW_COPY_AND_ASSIGN(GpuDevice);
  ~GpuDevice();
  MemType GetMemType() const override;
  std::string Name() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  void PreExecute() override;
  void Barrier(int) override;
  void DoCopyRemoteData(float*, float*, size_t, int) override;
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) override;
};
#endif

#ifdef HAS_MPI
class MpiDevice : public ThreadedDevice {
 public:
  MpiDevice( int rank, uint64_t device_id, DeviceListener*, int gpu_id);
  DISALLOW_COPY_AND_ASSIGN(MpiDevice);
  ~MpiDevice();
  MemType GetMemType() const override;
  std::string Name() const override;

 private:
  static size_t constexpr kParallelism = 4;
  int _gpu_id;
  void DoCopyRemoteData(float*, float*, size_t, int) override;
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) override;
  void DoExecute(Task& task, int thrid);
};
#endif

class CpuDevice : public ThreadedDevice {
 public:
  CpuDevice(uint64_t device_id, DeviceListener*);
#ifdef HAS_MPI
  CpuDevice(int rank, uint64_t device_id, DeviceListener*);
#endif
  DISALLOW_COPY_AND_ASSIGN(CpuDevice);
  ~CpuDevice();
  MemType GetMemType() const override;
  std::string Name() const override;

 private:
  static size_t constexpr kDefaultThreadNum = 4;
  void DoCopyRemoteData(float*, float*, size_t, int) override;
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) override;
};

}  // namespace minerva

