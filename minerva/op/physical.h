#pragma once
#include <memory>
#include "common/common.h"
#include "common/scale.h"
#include "op/compute_fn.h"

namespace minerva {


class PhysicalData: public Serializable {
public:
  PhysicalData(const Scale& s, uint64_t d, uint64_t id) : size(s), device_id(d), data_id(id) {
  }
  int GetSerializedSize() const override;
  int Serialize(char* buffer)  const  override ;
  static PhysicalData& DeSerialize(char*, int*)  ;
  Scale size;
  uint64_t device_id;
  uint64_t data_id;
  int extern_rc = 0;
#ifdef HAS_MPI
  PhysicalData(const Scale& s, int r, uint64_t d, uint64_t id) : size(s), device_id(d), data_id(id), rank(r) {
  }
  int rank;
#endif
};


class PhysicalOp: public Serializable {
public:
  PhysicalOp(): device_id(0){};
  PhysicalOp(std::shared_ptr<ComputeFn> fn, uint64_t id) : compute_fn(fn), device_id(id){};
  int GetSerializedSize()  const  override;
  int Serialize(char* buffer)  const override;
  static PhysicalOp& DeSerialize(char*, int*)  ;
  std::shared_ptr<ComputeFn> compute_fn;
  uint64_t device_id;
};

}  // end of namespace minerva

