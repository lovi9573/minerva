#pragma once
#include <dmlc/logging.h>
#include "op/basic_fn.h"
#include "op/data_shard.h"
#include "op/context.h"

namespace minerva {

struct Context;
class Task;

class ComputeFn : public BasicFn {
 public:
  virtual void Execute(DataList const&, DataList const&, Context const&) = 0;
  int GetSerializedSize() const override {
	  LOG(FATAL) << "GetSerializedSize() not implemented by derived class";
	  return 0;
  };
  int Serialize(char* buffer) const override {
	  LOG(FATAL) << "Serialize() not implemented by derived class";
	  return 0;
  };
  static std::shared_ptr<ComputeFn> DeSerialize(char*, int*);
#ifdef HAS_MPI
  virtual void Execute(Task const&, Context const&)=0;
#endif
};

}  // namespace minerva

