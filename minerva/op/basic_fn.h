#pragma once
#include <string>
#include "common/common.h"


namespace minerva {

class BasicFn : public Serializable  {
 public:
  virtual std::string Name() const = 0;
  virtual ~BasicFn() {}

};

template<class T>
struct ClosureTrait {
 public:
  T closure;
};

}
