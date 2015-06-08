#pragma once
#include <memory>
#include "common/scale.h"

namespace minerva {

class ComputeFn;

struct PhysicalData {
  PhysicalData(const Scale& s, uint64_t d, uint64_t id) : size(s), device_id(d), data_id(id) {
  }
  Scale size;
#ifdef HAS_MPI
  PhysicalData(const Scale& s, int r, uint64_t d, uint64_t id) : rank(r), size(s), device_id(d), data_id(id) {
  }
  int rank;
#endif
  uint64_t device_id;
  uint64_t data_id;
  int extern_rc = 0;
};

struct PhysicalOp {
  std::shared_ptr<ComputeFn> compute_fn;
  uint64_t device_id;
};

}  // end of namespace minerva

