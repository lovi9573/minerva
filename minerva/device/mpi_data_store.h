#pragma once
#include <queue>
#include "device/data_store.h"

namespace minerva {

/*
 * Maintains a symbolic store of allocated data on foreign mpi host.
 */
class MpiDataStore final : public DataStore {
 public:
  MpiDataStore(std::function<void*(size_t)> a, std::function<void(void*)> d);
  DISALLOW_COPY_AND_ASSIGN(MpiDataStore);
  virtual ~MpiDataStore();
  float* CreateData(uint64_t, size_t) override;
  void FreeData(uint64_t) override;
  size_t GetTotalBytes() const override;

  //TODO: May need a DataStore replacement specific to remote data.

 private:
  //std::unordered_map<size_t, std::queue<void*>> free_space_;
  //void ReleaseFreeSpace();
};

}  // namespace minerva

