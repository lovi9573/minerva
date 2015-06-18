#pragma once
#include <vector>
#include <cstdint>
#include "op/physical.h"
#include "device/task_data.h"

namespace minerva {

class Task : public Serializable{
public:
  Task(): id(0) {};
  int GetSerializedSize() const override;
  int Serialize(char* ) const override;
  static Task& DeSerialize(char* , int*)  ;
  std::vector<TaskData> inputs;
  std::vector<TaskData> outputs;
  PhysicalOp op;
  // `id` is only meaningful to the issuer of the task
  uint64_t id;
  // is this a light weight op? light weight op will be executed by the 
  // main thread to avoid thread switching
  bool light = false;
};

}  // namespace minerva
