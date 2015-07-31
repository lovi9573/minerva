#pragma once

#include <string>

namespace minerva {
  void PushGradAndPullWeight(const element_t * grad, element_t * weights, size_t size, const std::string & layer_name);
}