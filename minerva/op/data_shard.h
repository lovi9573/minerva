#pragma once
#include <vector>
#include "common/scale.h"

namespace minerva {

struct DataShard {
  DataShard(element_t* data, Scale const& size) : data_(data), size_(size) {
  }
  element_t* const data_;
  Scale const& size_;
};

using DataList = std::vector<DataShard>;

}  // namespace minerva

