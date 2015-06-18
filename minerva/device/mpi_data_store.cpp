#include "device/pooled_data_store.h"
#include "device/mpi_data_store.h"

using namespace std;

namespace minerva {

MpiDataStore::MpiDataStore(function<void*(size_t)> a, function<void(void*)> d) : DataStore(a, d) {
}

MpiDataStore::~MpiDataStore() {
  for (auto& i : data_states_) {
	deallocator_(i.second.ptr);
  }
}

//TODO: the DataStore.ptr pointer is really a unique ID # in the mpi context
float* MpiDataStore::CreateData(uint64_t id, size_t length) {
  lock_guard<mutex> lck(access_mutex_);
  DLOG(INFO) << "create mpi data #" << id << " length " << length;
  auto it = data_states_.emplace(id, DataState());
  CHECK(it.second) << "data already existed";
  auto& ds = it.first->second;
  ds.length = length;
  ds.ptr = allocator_(length);
  return static_cast<float*>(ds.ptr);
}

void MpiDataStore::FreeData(uint64_t id) {
  lock_guard<mutex> lck(access_mutex_);
  auto& ds = data_states_.at(id);
  deallocator_(ds.ptr);
  CHECK_EQ(data_states_.erase(id), 1);
}

size_t MpiDataStore::GetTotalBytes() const {
  lock_guard<mutex> lck(access_mutex_);
  size_t total_bytes = 0;
  for (auto& it : data_states_) {
	total_bytes += it.second.length;
  }
  return total_bytes;
}


}  // namespace minerva

