#pragma once

namespace minerva {

struct ConvInfo {
  ConvInfo(int ph = 0, int pw = 0, int sv = 1, int sh = 1)
    : pad_height(ph)
    , pad_width(pw)
    , stride_vertical(sv)
    , stride_horizontal(sh) {
  }
  int pad_height;
  int pad_width;
  int stride_vertical;
  int stride_horizontal;
};

struct PoolingInfo {
  enum class Algorithm : int {
    kMax,
    kAverage
  };
  PoolingInfo(
      Algorithm alg = Algorithm::kMax
    , int h = 0
    , int w = 0
    , int sv = 1
    , int sh = 1
    , int ph = 0
    , int pw = 0)
    : algorithm(alg)
    , height(h)
    , width(w)
    , stride_vertical(sv)
    , stride_horizontal(sh)
    , pad_height(ph)
    , pad_width(pw) {
  }
  Algorithm algorithm;
  int height;
  int width;
  int stride_vertical;
  int stride_horizontal;
  int pad_height;
  int pad_width;
};

enum class SoftmaxAlgorithm : int {
  kInstance,
  kChannel
};

enum class ActivationAlgorithm : int {
  kSigmoid,
  kRelu,
  kTanh
};

}  // namespace minerva

