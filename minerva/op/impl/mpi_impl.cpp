
#include "op/closure.h"
#include "mpi_impl.h"
#include "system/minerva_system.h"
//#include "../../mpi/mpi_server.h"
#include "../physical_op.h"
#include "op/impl/bundle.h"

namespace minerva {
#ifdef HAS_MPI
namespace mpi {

#define MPI_SIMPLE_CLOSURE_CALL(CLOSURE) \
		  /*CHECK_EQ(inputs.size(), 1) << "(" #CLOSURE ") #inputs is wrong!"; \
		  CHECK_EQ(outputs.size(), 1) << "(" #CLOSURE ") #outputs is wrong!";*/ \
		  MinervaSystem::Instance().mpi_server().MPI_Send_task(task, ctx); \

void Arithmetic(const Task& task, ArithmeticClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ArithmeticClosure);
}
void MatMult(const Task& task, MatMultClosure& closure, const Context& ctx){
	  MPI_SIMPLE_CLOSURE_CALL(MatMultClosure);
}
void ArithmeticConst(const Task& task, ArithmeticConstClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ArithmeticConstClosure);
}
void Transpose(const Task& task, TransposeClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(TransposeClosure);
}
//TODO: 4 Continue here translating each of these nested struct MPI calls.
void NormArithmetic(const Task& task, NormArithmeticClosure& closure, const Context &){
	NO_IMPL();
}
void Reduction(const Task& task, ReductionClosure& closure, const Context& ctx){
	NO_IMPL();
}
void MaxIndex(const Task& task, MaxIndexClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(MaxIndexClosure);
}
void Reshape(const Task& task, ReshapeClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ReshapeClosure);
}
void Elewise(const Task& task,  ElewiseClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ElewiseClosure);
}
void SigmoidForward(const Task& task, SigmoidForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(SigmoidForwardClosure);
}
void SigmoidBackward(const Task& task, SigmoidBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(SigmoidBackwardClosure);
}
void ReluForward(const Task& task, ReluForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ReluForwardClosure);
}
void ReluBackward(const Task& task, ReluBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ReluBackwardClosure);
}
void TanhForward(const Task& task, TanhForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(TanhForwardClosure);
}
void TanhBackward(const Task& task, TanhBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(TanhBackwardClosure);
}
void ConvForward(const Task& task, ConvForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ConvForwardClosure);
}
void ConvBackwardData(const Task& task, ConvBackwardDataClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ConvBackwardDataClosure);
}
void ConvBackwardFilter(const Task& task, ConvBackwardFilterClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ConvBackwardFilterClosure);
}
void ConvBackwardBias(const Task& task, ConvBackwardBiasClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ConvBackwardBiasClosure);
}
void SoftmaxForward(const Task& task, SoftmaxForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(SoftmaxForwardClosure);
}
void SoftmaxBackward(const Task& task, SoftmaxBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(SoftmaxBackwardClosure);
}
void ActivationForward(const Task& task, ActivationForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ActivationForwardClosure);
}
void ActivationBackward(const Task& task, ActivationBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ActivationBackwardClosure);
}
void PoolingForward(const Task& task, PoolingForwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(PoolingForwardClosure);
}
void PoolingBackward(const Task& task, PoolingBackwardClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(PoolingBackwardClosure);
}
void SyncWithPS(const Task& task,  SyncWithPSClosure& closure, const Context&){
	NO_IMPL();
}

void ArrayLoader(const Task& task,  ArrayLoaderClosure& closure, const Context& ctx){
	NO_IMPL();
}
void Randn(const Task& task,  RandnClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(RandnClosure);
}
void RandBernoulli(const Task& task,  RandBernoulliClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(RandBernoulliClosure);
}
void Fill(const Task& task,  FillClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(FillClosure);
}

void LRNForward(const Task& task, LRNForwardClosure& closure, const Context& ctx){
	NO_IMPL();
}
void LRNBackward(const Task& task, LRNBackwardClosure& closure, const Context& ctx){
	NO_IMPL();
}
void Concat(const Task& task, ConcatClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(ConcatClosure);
}
void Slice(const Task& task, SliceClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(SliceClosure);
}
void Index(const Task& task, IndexClosure& closure, const Context& ctx){
	MPI_SIMPLE_CLOSURE_CALL(IndexClosure);
}

void Select(const Task& task, SelectClosure&, const Context& ctx){
	NO_IMPL();
}
}  // end of namespace mpi
#endif
}  // end of namespace minerva
