#pragma once
#include "op/physical.h"
#include "op/closure.h"
#include "op/impl/bundle.h"
#include <sstream>
#include <vector>
#include <dmlc/logging.h>
#include "physical_fn.h"

namespace minerva {

#define SERIALIZABLE_OVERRIDE \
	    int GetSerializedSize() const ; \
	    int Serialize(char*) const ; \
		static std::shared_ptr<ComputeFn> DeSerialize(char*, int*);



// Data generate functions

class ArrayLoaderOp : public PhyDataGenFnWithClosure<ArrayLoaderClosure> {
 public:
  std::string Name() const {
    return ":array loader";
  }
  SERIALIZABLE_OVERRIDE
};

class RandnOp : public PhyDataGenFnWithClosure<RandnClosure> {
 public:
  std::string Name() const {
    return ":normal";
  }
  SERIALIZABLE_OVERRIDE
};

class RandBernoulliOp : public PhyDataGenFnWithClosure<RandBernoulliClosure> {
 public:
  std::string Name() const {
    return ":bernoulli";
  }
  SERIALIZABLE_OVERRIDE
};

class FillOp : public PhyDataGenFnWithClosure<FillClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << ":const=" << closure.val;
    return ss.str();
  }
  SERIALIZABLE_OVERRIDE
};

// Compute functions
class SyncWithPSOp : public ComputeFnWithClosure<SyncWithPSClosure> {
public:
  std::string Name() const {
    return std::string(":sync with ps on layer") + closure.layer_name;
  }
  SERIALIZABLE_OVERRIDE
};

class MatMultOp : public ComputeFnWithClosure<MatMultClosure> {
 public:
  std::string Name() const {
    return "*";
  }
  SERIALIZABLE_OVERRIDE
};

class TransOp : public ComputeFnWithClosure<TransposeClosure> {
 public:
  std::string Name() const {
    return "trans";
  }
  SERIALIZABLE_OVERRIDE
};

class ReductionOp : public ComputeFnWithClosure<ReductionClosure> {
 public:
  std::string Name() const {
   switch (closure.type) {
     case ReductionType::kSum:
       return "sum";
     case ReductionType::kMax:
       return "max";
   }
   return "reduction N/A";
  }
  SERIALIZABLE_OVERRIDE
};

class MaxIndexOp : public ComputeFnWithClosure<MaxIndexClosure> {
 public:
  std::string Name() const {
    return "max index";
  }
  SERIALIZABLE_OVERRIDE
};

class ReshapeOp : public ComputeFnWithClosure<ReshapeClosure> {
 public:
  std::string Name() const {
    return "reshape";
  }
  SERIALIZABLE_OVERRIDE
};

class ElewiseOp : public ComputeFnWithClosure<ElewiseClosure> {
 public:
  std::string Name() const {
    switch(closure.type) {
      case ElewiseType::kExp:      return "exp";
      case ElewiseType::kLn:       return "ln";
      case ElewiseType::kNegative: return "-";
    };
    return "NA";
  }
  SERIALIZABLE_OVERRIDE
};

class ArithmeticOp : public ComputeFnWithClosure<ArithmeticClosure> {
 public:
  std::string Name() const {
    switch(closure.type) {
      case ArithmeticType::kAdd:   return "+";
      case ArithmeticType::kSub:   return "-";
      case ArithmeticType::kMult:  return ".*";
      case ArithmeticType::kDiv:   return "./";
    };
    return "NA";
  }
  SERIALIZABLE_OVERRIDE
};

class ArithmeticConstOp : public ComputeFnWithClosure<ArithmeticConstClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    if(closure.side == 0) { // left
      ss << closure.val;
    }
    switch(closure.type) {
      case ArithmeticType::kAdd:   ss << "+."; break;
      case ArithmeticType::kSub:   ss << "-."; break;
      case ArithmeticType::kMult:  ss << ".*"; break;
      case ArithmeticType::kDiv:   ss << "./"; break;
    };
    if(closure.side == 1) { // right
      ss << closure.val;
    }
    return ss.str();
  }
  SERIALIZABLE_OVERRIDE
};

class NormArithmeticOp : public ComputeFnWithClosure<NormArithmeticClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    switch (closure.type) {
      case ArithmeticType::kAdd:
        ss << "+";
        break;
      case ArithmeticType::kSub:
        ss << "-";
        break;
      case ArithmeticType::kMult:
        ss << ".*";
        break;
      case ArithmeticType::kDiv:
        ss << "./";
        break;
    }
    ss << " norm";
    return ss.str();
  }
  SERIALIZABLE_OVERRIDE
};

class SigmoidForwardOp : public ComputeFnWithClosure<SigmoidForwardClosure> {
 public:
  std::string Name() const {
    return "sigmoid forward";
  }
  SERIALIZABLE_OVERRIDE
};

class SigmoidBackwardOp : public ComputeFnWithClosure<SigmoidBackwardClosure> {
 public:
  std::string Name() const {
    return "sigmoid backward";
  }
  SERIALIZABLE_OVERRIDE
};

class ThresholdNormOp : public ComputeFnWithClosure<ThresholdNormClosure> {
 public:
  std::string Name() const {
    return "threshold norm";
  }
};

class ReluForwardOp : public ComputeFnWithClosure<ReluForwardClosure> {
 public:
  std::string Name() const {
    return "relu forward";
  }
  SERIALIZABLE_OVERRIDE
};

class ReluBackwardOp : public ComputeFnWithClosure<ReluBackwardClosure> {
 public:
  std::string Name() const {
    return "relu backward";
  }
  SERIALIZABLE_OVERRIDE
};

class TanhForwardOp : public ComputeFnWithClosure<TanhForwardClosure> {
 public:
  std::string Name() const {
    return "tanh forward";
  }
  SERIALIZABLE_OVERRIDE
};

class TanhBackwardOp : public ComputeFnWithClosure<TanhBackwardClosure> {
 public:
  std::string Name() const {
    return "tanh backward";
  }
  SERIALIZABLE_OVERRIDE
};

class ConvForwardOp : public ComputeFnWithClosure<ConvForwardClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << "pad:" << closure.pad_height << "*" << closure.pad_width;
    ss << " stride:" << closure.stride_vertical << "*" << closure.stride_horizontal;
    ss << " conv ff";
    return ss.str();
  }
  SERIALIZABLE_OVERRIDE
};

class ConvBackwardDataOp : public ComputeFnWithClosure<ConvBackwardDataClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << "pad:" << closure.pad_height << "*" << closure.pad_width;
    ss << " stride:" << closure.stride_vertical << "*" << closure.stride_horizontal;
    ss << " conv bp data";
    return ss.str();
  }
  SERIALIZABLE_OVERRIDE
};

class ConvBackwardFilterOp : public ComputeFnWithClosure<ConvBackwardFilterClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << "pad:" << closure.pad_height << "*" << closure.pad_width;
    ss << " stride:" << closure.stride_vertical << "*" << closure.stride_horizontal;
    ss << " conv bp filter";
    return ss.str();
  }
  SERIALIZABLE_OVERRIDE
};

class ConvBackwardBiasOp : public ComputeFnWithClosure<ConvBackwardBiasClosure> {
 public:
  std::string Name() const {
    return "conv bp bias";
  }
  SERIALIZABLE_OVERRIDE
};

class SoftmaxForwardOp : public ComputeFnWithClosure<SoftmaxForwardClosure> {
 public:
  std::string Name() const {
    switch (closure.algorithm) {
      case SoftmaxAlgorithm::kInstance:
        return "instance softmax ff";
      case SoftmaxAlgorithm::kChannel:
        return "channel softmax ff";
    }
    return "unknown softmax ff";
  }
  SERIALIZABLE_OVERRIDE
};

class SoftmaxBackwardOp : public ComputeFnWithClosure<SoftmaxBackwardClosure> {
 public:
  std::string Name() const {
    switch (closure.algorithm) {
      case SoftmaxAlgorithm::kInstance:
        return "instance softmax bp";
      case SoftmaxAlgorithm::kChannel:
        return "channel softmax bp";
    }
    return "unknown softmax bp";
  }
  SERIALIZABLE_OVERRIDE
};

class ActivationForwardOp : public ComputeFnWithClosure<ActivationForwardClosure> {
 public:
  std::string Name() const {
    switch (closure.algorithm) {
      case ActivationAlgorithm::kSigmoid:
        return "sigmoid ff";
      case ActivationAlgorithm::kRelu:
        return "relu ff";
      case ActivationAlgorithm::kTanh:
        return "tanh ff";
    }
    return "unknown activation ff";
  }
  SERIALIZABLE_OVERRIDE
};

class ActivationBackwardOp : public ComputeFnWithClosure<ActivationBackwardClosure> {
 public:
  std::string Name() const {
    switch (closure.algorithm) {
      case ActivationAlgorithm::kSigmoid:
        return "sigmoid bp";
      case ActivationAlgorithm::kRelu:
        return "relu bp";
      case ActivationAlgorithm::kTanh:
        return "tanh bp";
    }
    return "unknown activation bp";
  }
  SERIALIZABLE_OVERRIDE
};

class PoolingForwardOp : public ComputeFnWithClosure<PoolingForwardClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    switch (closure.algorithm) {
      case PoolingInfo::Algorithm::kMax:
        ss << "max pooling ff";
        break;
      case PoolingInfo::Algorithm::kAverage:
        ss << "average pooling ff";
        break;
    }
    ss << " " << closure.height << "*" << closure.width;
    ss << " stride:" << closure.stride_horizontal << "*" << closure.stride_vertical;
    return ss.str();
  }
  SERIALIZABLE_OVERRIDE
};

class PoolingBackwardOp : public ComputeFnWithClosure<PoolingBackwardClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    switch (closure.algorithm) {
      case PoolingInfo::Algorithm::kMax:
        ss << "max pooling bp";
        break;
      case PoolingInfo::Algorithm::kAverage:
        ss << "average pooling bp";
        break;
    }
    ss << " " << closure.height << "*" << closure.width;
    ss << " stride:" << closure.stride_horizontal << "*" << closure.stride_vertical;
    return ss.str();
  }
  SERIALIZABLE_OVERRIDE
};

class LRNForwardOp : public ComputeFnWithClosure<LRNForwardClosure> {
 public:
  std::string Name() const {
    return "LRN Forward";
  }
  SERIALIZABLE_OVERRIDE
};

class LRNBackwardOp : public ComputeFnWithClosure<LRNBackwardClosure> {
 public:
  std::string Name() const {
    return "LRN Backward";
  }
  SERIALIZABLE_OVERRIDE
};

class ConcatOp : public ComputeFnWithClosure<ConcatClosure> {
 public:
  std::string Name() const {
    return "Concat";
  }
  SERIALIZABLE_OVERRIDE
};

class SliceOp : public ComputeFnWithClosure<SliceClosure> {
 public:
  std::string Name() const {
    return "Slice";
  }
  SERIALIZABLE_OVERRIDE
};

class IndexOp : public ComputeFnWithClosure<IndexClosure> {
 public:
  std::string Name() const {
    return "Index";
  }
  SERIALIZABLE_OVERRIDE
};

class SelectOp : public ComputeFnWithClosure<SelectClosure> {
 public:
  std::string Name() const {
    return "Select";
  }
  SERIALIZABLE_OVERRIDE
};

}  // namespace minerva

