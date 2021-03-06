#pragma once
#include <memory>
#include <cstdlib>
#include <dmlc/logging.h>
#include <stdio.h>

namespace minerva {
namespace common {

template<typename T>
class EverlastingSingleton {
 public:
  static T& Instance() {
	  //TODO(jesselovitt): Got one of the below errors at the natural end of an mpi run.
    CHECK(data_) << "please initialize before use";
    return *data_;
  }
  static void Initialize(int* argc, char*** argv) {
    CHECK(!data_) << "already initialized";
    data_.reset(new T(argc, argv));
    atexit(Finalize);
  }
  static bool IsAlive() {
    return static_cast<bool>(data_);
  }
 private:
  static std::unique_ptr<T> data_;
  static void Finalize() {
    CHECK(data_) << "not alive";
    data_.reset();
  }

};

# if defined(_MSC_VER)
template<typename T> std::unique_ptr<T> EverlastingSingleton<T>::data_ = nullptr;
# else
template<typename T> std::unique_ptr<T> EverlastingSingleton<T>::data_{};
# endif

}  // namespace common
}  // namespace minerva
