#pragma once
#include "op/impl/basic.h"
#include "op/impl/cuda.h"
#include "op/impl/fpga.h"
#include "op/impl/impl.h"
#include <dmlc/logging.h>

namespace minerva {

template<typename... Args>
void NO_IMPL(Args&&...) {
  LOG(FATAL) << "no implementation";
}

template<typename I, typename O, typename C, typename... Args>
void NO_IMPL(const I & i, const O & o, C & c, Args&&...) {
  LOG(FATAL) << "no implementation for " << typeid(C).name();
}
//TODO(jlovitt): 1 Replace fpga NO_IMPL's with proper function calls as implemented.


INSTALL_COMPUTE_FN(ArithmeticClosure, 			basic::Arithmetic, 			NO_IMPL, cuda::Arithmetic, NO_IMPL);
INSTALL_COMPUTE_FN(ArithmeticConstClosure, 		basic::ArithmeticConst, 	NO_IMPL, cuda::ArithmeticConst, NO_IMPL);
INSTALL_COMPUTE_FN(MatMultClosure, 				basic::MatMult, 			NO_IMPL, cuda::MatMult, NO_IMPL);
INSTALL_COMPUTE_FN(TransposeClosure, 			basic::Transpose, 			NO_IMPL, cuda::Transpose, NO_IMPL);
INSTALL_COMPUTE_FN(ReductionClosure, 			basic::Reduction, 			NO_IMPL, cuda::Reduction, NO_IMPL);
INSTALL_COMPUTE_FN(NormArithmeticClosure, 		basic::NormArithmetic, 		NO_IMPL, cuda::NormArithmetic, NO_IMPL);
INSTALL_COMPUTE_FN(MaxIndexClosure, 			basic::MaxIndex, 			NO_IMPL, cuda::MaxIndex, NO_IMPL);
INSTALL_COMPUTE_FN(ReshapeClosure, 				basic::Reshape, 			NO_IMPL, cuda::Reshape, NO_IMPL);
INSTALL_COMPUTE_FN(ElewiseClosure, 				basic::Elewise, 			NO_IMPL, cuda::Elewise, NO_IMPL);
INSTALL_COMPUTE_FN(ThresholdNormClosure, 		basic::ThresholdNorm,		NO_IMPL, NO_IMPL,			NO_IMPL);
INSTALL_COMPUTE_FN(SigmoidForwardClosure, 		basic::SigmoidForward, 		NO_IMPL, cuda::SigmoidForward, NO_IMPL);
INSTALL_COMPUTE_FN(SigmoidBackwardClosure, 		NO_IMPL, 					NO_IMPL, cuda::SigmoidBackward, NO_IMPL);
INSTALL_COMPUTE_FN(ReluForwardClosure, 			basic::ReluForward, 		NO_IMPL, cuda::ReluForward, fpga::ReluForward);
INSTALL_COMPUTE_FN(ReluBackwardClosure, 		basic::ReluBackward, 		NO_IMPL, cuda::ReluBackward, NO_IMPL);
INSTALL_COMPUTE_FN(TanhForwardClosure, 			basic::TanhForward, 		NO_IMPL, cuda::TanhForward, NO_IMPL);
INSTALL_COMPUTE_FN(TanhBackwardClosure, 		NO_IMPL, 					NO_IMPL, cuda::TanhBackward, NO_IMPL);
INSTALL_COMPUTE_FN(ConvForwardClosure, 			basic::ConvForward, 		NO_IMPL, cuda::ConvForward, fpga::ConvForward);
INSTALL_COMPUTE_FN(ConvBackwardDataClosure, 	basic::ConvBackwardData,	NO_IMPL, cuda::ConvBackwardData, NO_IMPL);
INSTALL_COMPUTE_FN(ConvBackwardFilterClosure, 	basic::ConvBackwardFilter, 	NO_IMPL, cuda::ConvBackwardFilter, 	NO_IMPL);
INSTALL_COMPUTE_FN(ConvBackwardBiasClosure, 	basic::ConvBackwardBias, 	NO_IMPL, cuda::ConvBackwardBias, 	NO_IMPL);
INSTALL_COMPUTE_FN(SoftmaxForwardClosure, 		basic::SoftmaxForward,		NO_IMPL, cuda::SoftmaxForward, 		NO_IMPL);
INSTALL_COMPUTE_FN(SoftmaxBackwardClosure, 		NO_IMPL, 					NO_IMPL, cuda::SoftmaxBackward, 	NO_IMPL);
INSTALL_COMPUTE_FN(ActivationForwardClosure, 	basic::ActivationForward, 	NO_IMPL, cuda::ActivationForward, NO_IMPL);
INSTALL_COMPUTE_FN(ActivationBackwardClosure, 	NO_IMPL, 					NO_IMPL, cuda::ActivationBackward, NO_IMPL);
INSTALL_COMPUTE_FN(PoolingForwardClosure, 		basic::PoolingForward, 		NO_IMPL, cuda::PoolingForward, NO_IMPL);
INSTALL_COMPUTE_FN(PoolingBackwardClosure,  	basic::PoolingBackward, 	NO_IMPL, cuda::PoolingBackward, NO_IMPL);
INSTALL_COMPUTE_FN(SyncWithPSClosure, 			basic::SyncWithPS,			NO_IMPL, cuda::SyncWithPS, NO_IMPL);
INSTALL_DATAGEN_FN(ArrayLoaderClosure, 			basic::ArrayLoader, 		NO_IMPL, cuda::ArrayLoader, NO_IMPL);
INSTALL_DATAGEN_FN(RandnClosure, 				basic::Randn, 				NO_IMPL, cuda::Randn, NO_IMPL);
INSTALL_DATAGEN_FN(RandBernoulliClosure, 		basic::RandBernoulli, 		NO_IMPL, cuda::RandBernoulli, NO_IMPL);
INSTALL_DATAGEN_FN(FillClosure, 				basic::Fill, 				NO_IMPL, cuda::Fill, 				NO_IMPL);
INSTALL_COMPUTE_FN(LRNForwardClosure, 			basic::LRNForward, 			NO_IMPL, cuda::LRNForward, NO_IMPL);
INSTALL_COMPUTE_FN(LRNBackwardClosure, 			basic::LRNBackward, 		NO_IMPL, cuda::LRNBackward, NO_IMPL);
INSTALL_COMPUTE_FN(ConcatClosure, 				NO_IMPL, 					NO_IMPL, cuda::Concat, 		NO_IMPL);
INSTALL_COMPUTE_FN(SliceClosure, 				NO_IMPL, 					NO_IMPL, cuda::Slice, 		NO_IMPL);
INSTALL_COMPUTE_FN(IndexClosure, 				basic::Index, 				NO_IMPL, NO_IMPL, 			NO_IMPL);
INSTALL_COMPUTE_FN(SelectClosure,	 			NO_IMPL, 					NO_IMPL, cuda::Select, 		NO_IMPL);
INSTALL_COMPUTE_FN(HistogramClosure,	 		basic::Histogram, 			NO_IMPL, NO_IMPL, 			NO_IMPL);
INSTALL_DATAGEN_FN(RandUniformClosure,	 		basic::RandUniform, 		NO_IMPL, NO_IMPL, 			NO_IMPL);



}  // namespace minerva
