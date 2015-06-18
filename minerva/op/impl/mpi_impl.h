#pragma once
#include "../physical_fn.h"
#include "op/closure.h"

namespace minerva {
#ifdef HAS_MPI
namespace mpi {

void Arithmetic(const Task&, ArithmeticClosure&, const Context&);
void MatMult(const Task&, MatMultClosure&, const Context&);
void ArithmeticConst(const Task&, ArithmeticConstClosure&, const Context&);
void Transpose(const Task&, TransposeClosure&, const Context&);
void NormArithmetic(const Task&, NormArithmeticClosure&, const Context &);
void Reduction(const Task&, ReductionClosure&, const Context&);
void MaxIndex(const Task&, MaxIndexClosure&, const Context&);
void Reshape(const Task&, ReshapeClosure&, const Context&);
void Elewise(const Task&, ElewiseClosure&, const Context&);
void SigmoidForward(const Task&, SigmoidForwardClosure&, const Context&);
void SigmoidBackward(const Task&, SigmoidBackwardClosure&, const Context&);
void ReluForward(const Task&, ReluForwardClosure&, const Context&);
void ReluBackward(const Task&, ReluBackwardClosure&, const Context&);
void TanhForward(const Task&, TanhForwardClosure&, const Context&);
void TanhBackward(const Task&, TanhBackwardClosure&, const Context&);
void ConvForward(const Task&, ConvForwardClosure&, const Context&);
void ConvBackwardData(const Task&, ConvBackwardDataClosure&, const Context&);
void ConvBackwardFilter(const Task&, ConvBackwardFilterClosure&, const Context&);
void ConvBackwardBias(const Task&, ConvBackwardBiasClosure&, const Context&);
void SoftmaxForward(const Task&, SoftmaxForwardClosure&, const Context&);
void SoftmaxBackward(const Task&, SoftmaxBackwardClosure&, const Context&);
void ActivationForward(const Task&, ActivationForwardClosure&, const Context&);
void ActivationBackward(const Task&, ActivationBackwardClosure&, const Context&);
void PoolingForward(const Task&, PoolingForwardClosure&, const Context&);
void PoolingBackward(const Task&, PoolingBackwardClosure&, const Context&);
void SyncWithPS(const Task&, SyncWithPSClosure& closure, const Context&);

void ArrayLoader(const Task&, ArrayLoaderClosure& closure, const Context&);
void Randn(const Task&, RandnClosure&, const Context&);
void RandBernoulli(const Task&, RandBernoulliClosure&, const Context&);
void Fill(const Task&, FillClosure&, const Context&);

void LRNForward(const Task&, LRNForwardClosure&, const Context&);
void LRNBackward(const Task&, LRNBackwardClosure&, const Context&);
void Concat(const Task&, ConcatClosure&, const Context&);
void Slice(const Task&, SliceClosure&, const Context&);
void Index(const Task&, IndexClosure&, const Context&);
void Select(const Task&, SelectClosure&, const Context&);

}  // end of namespace mpi
#endif
}  // end of namespace minerva
