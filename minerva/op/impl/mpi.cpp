#pragma once
#include "op/physical_fn.h"
#include "op/closure.h"

namespace minerva {
#ifdef HAS_MPI
namespace mpi {

#define MPI_SIMPLE_CLOSURE_CALL(closure, closurecode) \
		  CHECK_EQ(inputs.size(), 1) << "(" #closure ") #inputs is wrong!"; \
		  CHECK_EQ(outputs.size(), 1) << "(" #closure ") #outputs is wrong!"; \
		  MPI::MPI_Send_task_desciptor(task,closurecode, ctx); \
		  MPI::MPI_Send_task_inputs(task,ctx); \
		  MPI::MPI_Send_task_outputs(task,ctx); \
		  mpi_send(&closure, sizeof(closure),MPI_BYTE, ctx.rank, MPI_CLOSURE, MPI_COMM_WORLD);

void Arithmetic(const Task& task, ArithmeticClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ArithmeticClosure,ARITHMETICCLOSURE);
}
void MatMult(const Task& task, MatMultClosure& closure, const Context& ctx){
	  MPI_SIMPLE_CLOSURE_CALL(MatMultClosure,MATMULTCLOSURE);
}
void ArithmeticConst(const Task& task, ArithmeticConstClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ArithmeticConstClosure,ARITHMETICCONSTCLOSURE);
}
void Transpose(const Task& task, TransposeClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(TransposeClosure,TRANSPOSECLOSURE);
}
//TODO: 4 Continue here translating each of these nested struct MPI calls.
void NormArithmetic(const Task& task, NormArithmeticClosure&, const Context &);
void Reduction(const Task& task, ReductionClosure& closure, const Context& ctx);
void MaxIndex(const Task& task, MaxIndexClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(MaxIndexClosure,MAXINDEXCLOSURE);
}
void Reshape(const Task& task, ReshapeClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ReshapeClosure,RESHAPECLOSURE);
}
void Elewise(const Task& task,  ElewiseClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ElewiseClosure,ELEWISECLOSURE);
}
void SigmoidForward(const Task& task, SigmoidForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(SigmoidForwardClosure,SIGMOIDFORWARDCLOSURE);
}
void SigmoidBackward(const Task& task, SigmoidBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(SigmoidBackwardClosure,SIGMOIDBACKWARDCLOSURE);
}
void ReluForward(const Task& task, ReluForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ReluForwardClosure,RELUFORWARDCLOSURE);
}
void ReluBackward(const Task& task, ReluBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ReluBackwardClosure,RELUBACKWARDCLOSURE);
}
void TanhForward(const Task& task, TanhForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ReluBackwardClosure,RELUBACKWARDCLOSURE);
}
void TanhBackward(const Task& task, TanhBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(TanhBackwardClosure,TANHBACKWARDCLOSURE);
}
void ConvForward(const Task& task, ConvForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ConvForwardClosure,CONVFORWARDCLOSURE);
}
void ConvBackwardData(const Task& task, ConvBackwardDataClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ConvBackwardDataClosure,CONVBACKWARDDATACLOSURE);
}
void ConvBackwardFilter(const Task& task, ConvBackwardFilterClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ConvBackwardFilterClosure,CONVBACKWARDFILTERCLOSURE);
}
void ConvBackwardBias(const Task& task, ConvBackwardBiasClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ConvBackwardBiasClosure,CONVBACKWARDBIASCLOSURE);
}
void SoftmaxForward(const Task& task, SoftmaxForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(SoftmaxForwardClosure,SOFTMAXFORWARDCLOSURE);
}
void SoftmaxBackward(const Task& task, SoftmaxBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(SoftmaxBackwardClosure,SOFTMAXBACKWARDCLOSURE);
}
void ActivationForward(const Task& task, ActivationForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ActivationForwardClosure,ACTIVATIONFORWARDCLOSURE);
}
void ActivationBackward(const Task& task, ActivationBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ActivationBackwardClosure,ACTIVATIONBACKWARDCLOSURE);
}
void PoolingForward(const Task& task, PoolingForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(PoolingForwardClosure,POOLINGFORWARDCLOSURE);
}
void PoolingBackward(const Task& task, PoolingBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(PoolingBackwardClosure,POOLINGBACKWARDCLOSURE);
}
void SyncWithPS(const Task& task,  SyncWithPSClosure& closure, const Context&);

void ArrayLoader(const Task& task,  ArrayLoaderClosure& closure, const Context& ctx);
void Randn(const Task& task,  RandnClosure&, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(RandnClosure,RANDNCLOSURE);
}
void RandBernoulli(const Task& task,  RandBernoulliClosure&, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(RandBernoulliClosure,RANDBERNOULLICLOSURE);
}
void Fill(const Task& task,  FillClosure&, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(FillClosure,FILLCLOSURE);
}

void LRNForward(const Task& task, LRNForwardClosure& closure, const Context& ctx);
void LRNBackward(const Task& task, LRNBackwardClosure& closure, const Context& ctx);
void Concat(const Task& task, ConcatClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ConcatClosure,CONCATCLOSURE);
}
void Slice(const Task& task, SliceClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(SliceClosure,SLICECLOSURE);
}
void Index(const Task& task, IndexClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(IndexClosure,INDEXCLOSURE);
}

void Select(const Task task, SelectClosure&, const Context& ctx);
}  // end of namespace mpi
#endif
}  // end of namespace minerva
