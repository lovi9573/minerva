#pragma once
#include <string>
#include <utility>
#include <mutex>
#include <memory>

#include "../op/physical_fn.h"
#include "device/task.h"
#include "device/data_store.h"
#include "device/device_listener.h"
#include "common/common.h"
#include "common/thread_pool.h"
#include "common/concurrent_blocking_queue.h"
#include "common/concurrent_unordered_set.h"
#include "common/concurrent_unordered_map.h"

#define DEFAULT_POOL_SIZE ((size_t) 5.8 * 1024 * 1024 * 1024)

namespace minerva {

/*
 * Base class for a device
 */
class Device {
 public:
  enum class MemType {
    kCpu,
    kGpu,
	kFpga,
    kMpi,
  };
  Device() = delete;
  DISALLOW_COPY_AND_ASSIGN(Device);
  Device(uint64_t device_id, DeviceListener*);
  virtual ~Device() = default;
  virtual void PushTask(Task*) = 0;
  virtual std::pair<MemType, element_t*> GetPtr(uint64_t data_id);
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
  std::unique_ptr<DataStore> data_store_;
  DeviceListener* listener_;
  int rank_;
};

class ThreadedDevice : public Device {
 public:
  ThreadedDevice() = delete;
  ThreadedDevice(uint64_t device_id, DeviceListener* l, size_t parallelism);
#ifdef HAS_MPI
  ThreadedDevice(int rank, uint64_t device_id, DeviceListener* l, size_t parallelism);
#endif
  DISALLOW_COPY_AND_ASSIGN(ThreadedDevice);
  ~ThreadedDevice() = default;
  void PushTask(Task*) override;
  void FreeDataIfExist(uint64_t data_id) override;

 protected:
  virtual void Execute(Task*, int thrid);
  virtual void PreExecute();
  virtual void Barrier(int);
  virtual void DoCopyRemoteData(element_t*, element_t*, size_t, int) = 0;
  virtual void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) = 0;
  virtual void DoExecute(Task* task, int thrid) = 0;
  ConcurrentUnorderedMap<uint64_t, std::mutex> copy_locks_;
  ThreadPool pool_;
};


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
# if defined(_MSC_VER)
    static const size_t kDefaultThreadNum = 4;
# else
    static size_t constexpr kDefaultThreadNum = 4;
# endif
  void DoCopyRemoteData(element_t*, element_t*, size_t, int) override;
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) override;
  void DoExecute(Task* task, int thrid) override;
};

}  // namespace minerva

