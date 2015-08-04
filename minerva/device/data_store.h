#pragma once
#include <unordered_map>
#include <functional>
#include <mutex>
#include <cstdint>
#include <cstddef>
#include <dmlc/logging.h>
#include "common/common.h"

namespace minerva {

class DataStore {
 public:
  DataStore(std::function<void*(size_t)> a, std::function<void(void*)> d);
  DISALLOW_COPY_AND_ASSIGN(DataStore);
  virtual ~DataStore();
  virtual element_t* CreateData(uint64_t, size_t);
  virtual element_t* GetData(uint64_t);
  virtual bool ExistData(uint64_t id) const;
  virtual void FreeData(uint64_t);
  virtual size_t GetTotalBytes() const;

 protected:
  struct DataState {
    void* ptr;
    size_t length;
  };
  mutable std::mutex access_mutex_;
  std::unordered_map<uint64_t, DataState> data_states_;
  std::function<void*(size_t)> allocator_;
  std::function<void(void*)> deallocator_;
};

}  // namespace minerva

