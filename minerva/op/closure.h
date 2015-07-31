#pragma once
#include <memory>
#include "common/scale.h"
#include "common/common.h"
#include "narray/convolution_info.h"


namespace minerva {


#define ARRAYLOADERCLOSURE  0
#define RANDNCLOSURE 1
#define RANDBERNOULLICLOSURE 2
#define FILLCLOSURE 3
#define SYNCWITHPSCLOSURE 4
#define MATMULTCLOSURE 5
#define TRANSPOSECLOSURE 6
#define RESHAPECLOSURE 7
#define REDUCTIONCLOSURE 8
#define MAXINDEXCLOSURE 9
#define ELEWISECLOSURE 10
#define SIGMOIDFORWARDCLOSURE 11
#define SIGMOIDBACKWARDCLOSURE 12
#define RELUFORWARDCLOSURE 13
#define RELUBACKWARDCLOSURE 14
#define TANHFORWARDCLOSURE 15
#define TANHBACKWARDCLOSURE 16
#define ARITHMETICCLOSURE 17
#define ARITHMETICCONSTCLOSURE 18
#define NORMARITHMETICCLOSURE 19
#define CONVCLOSURE 20
#define CONVFORWARDCLOSURE 21
#define CONVBACKWARDDATACLOSURE 22
#define CONVBACKWARDFILTERCLOSURE 23
#define CONVBACKWARDBIASCLOSURE 24
#define SOFTMAXCLOSURE 25
#define SOFTMAXFORWARDCLOSURE 26
#define SOFTMAXBACKWARDCLOSURE 27
#define ACTIVATIONCLOSURE 28
#define ACTIVATIONFORWARDCLOSURE 29
#define ACTIVATIONBACKWARDCLOSURE 30
#define POOLINGCLOSURE 31
#define POOLINGFORWARDCLOSURE 32
#define POOLINGBACKWARDCLOSURE 33
#define LRNCLOSURE 34
#define LRNFORWARDCLOSURE 35
#define LRNBACKWARDCLOSURE 36
#define CONCATCLOSURE 37
#define SLICECLOSURE 38
#define INDEXCLOSURE 39
#define SELECTCLOSURE 40


enum class ArithmeticType : int {
  kAdd = 0,
  kSub,
  kMult,
  kDiv,
};

enum class ElewiseType : int {
  kExp = 0,
  kLn,
  kNegative,
};

enum class ReductionType : int {
  kSum = 0,
  kMax,
};


struct ArrayLoaderClosure {
#ifdef HAS_MPI
  int count;
#endif
  std::shared_ptr<element_t> data;
};

struct RandnClosure {
  float mu, var;
};

struct RandBernoulliClosure {
  float p;
};

struct FillClosure {
  element_t val;
};

struct SyncWithPSClosure {
  std::string layer_name;
};

struct MatMultClosure {
};

struct TransposeClosure {
};

struct ReshapeClosure {
};

struct ReductionClosure {
  ReductionType type;
  Scale dims_to_reduce;
};

struct MaxIndexClosure {
  int dim;
};

struct ElewiseClosure {
  ElewiseType type;
};

struct SigmoidForwardClosure {
};

struct SigmoidBackwardClosure {
};

struct ReluForwardClosure {
};

struct ReluBackwardClosure {
};

struct TanhForwardClosure {
};

struct TanhBackwardClosure {
};

struct ArithmeticClosure {
  ArithmeticType type;
};

struct ArithmeticConstClosure {
  ArithmeticType type;
  element_t val;
  int side; // 0 is left const, 1 is right const
};

struct NormArithmeticClosure {
  ArithmeticType type;
  Scale dims_to_replicate;
};

template<int i> struct ConvClosure {
  int pad_height;
  int pad_width;
  int stride_vertical;
  int stride_horizontal;
};

typedef ConvClosure<0> ConvForwardClosure;

typedef ConvClosure<1> ConvBackwardDataClosure;

typedef ConvClosure<2> ConvBackwardFilterClosure;

struct ConvBackwardBiasClosure {
};

template<int i> struct SoftmaxClosure {
  SoftmaxAlgorithm algorithm;
};

typedef SoftmaxClosure<0> SoftmaxForwardClosure;

typedef SoftmaxClosure<1> SoftmaxBackwardClosure;

template<int i> struct ActivationClosure {
  ActivationAlgorithm algorithm;
};

typedef ActivationClosure<0> ActivationForwardClosure;

typedef ActivationClosure<1> ActivationBackwardClosure;

template<int i> struct PoolingClosure {
  PoolingInfo::Algorithm algorithm;
  int height;
  int width;
  int stride_vertical;
  int stride_horizontal;
  int pad_height;
  int pad_width;
};

typedef PoolingClosure<0> PoolingForwardClosure;

typedef PoolingClosure<1> PoolingBackwardClosure;

template<int i> struct LRNClosure {
	int local_size;
	element_t alpha, beta;
	Scale data_shape;
};

typedef LRNClosure<0> LRNForwardClosure;
typedef LRNClosure<1> LRNBackwardClosure;

struct ConcatClosure {
  int catdim;
};


struct SliceClosure {
  int slice_dim;
  int st_off;
  int slice_count;
};

struct IndexClosure {
  int idx;
};

struct SelectClosure {
  std::vector<int> indices;
};


}  // end of namespace minerva


