#pragma once
#include "narray/narray.h"

namespace minerva {

// TODO yutian: better use namespace
class Elewise {
 public:
  static NArray Mult(const NArray&, const NArray&);
  static NArray Exp(const NArray&);
  static NArray Ln(const NArray&);
  static NArray SigmoidForward(const NArray&);
  static NArray SigmoidBackward(const NArray& diff, const NArray& top, const NArray& bottom);
  static NArray ReluForward(const NArray&);
  static NArray ReluBackward(const NArray& diff, const NArray& top, const NArray& bottom);
  static NArray TanhForward(const NArray&);
  static NArray TanhBackward(const NArray& diff, const NArray& top, const NArray& bottom);
};

NArray operator+(const NArray&, const NArray&);
NArray operator-(const NArray&, const NArray&);
NArray operator/(const NArray&, const NArray&);
NArray operator+(element_t, const NArray&);
NArray operator-(element_t, const NArray&);
NArray operator*(element_t, const NArray&);
NArray operator/(element_t, const NArray&);
NArray operator+(const NArray&, element_t);
NArray operator-(const NArray&, element_t);
NArray operator*(const NArray&, element_t);
NArray operator/(const NArray&, element_t);

}

