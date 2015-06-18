/*
 * compute_fn.cpp
 *
 *  Created on: Jun 17, 2015
 *      Author: jlovitt
 */

#include "op/compute_fn.h"
#include "op/closure.h"
#include "op/physical_op.h"



namespace minerva {

#define CASE_CLOSURECODE_OP(CLOSURECODE,OP) \
		case CLOSURECODE: \
		return OP::DeSerialize(buffer, offset);



std::shared_ptr<ComputeFn> ComputeFn::DeSerialize(char* buffer, int* offset){
	int opcode = *(int*)(buffer+*offset);
	switch (opcode){
		CASE_CLOSURECODE_OP( SIGMOIDFORWARDCLOSURE,SigmoidForwardOp )
		CASE_CLOSURECODE_OP( TRANSPOSECLOSURE,TransOp )
		CASE_CLOSURECODE_OP( REDUCTIONCLOSURE,ReductionOp )
		CASE_CLOSURECODE_OP( ARRAYLOADERCLOSURE,ArrayLoaderOp )
		CASE_CLOSURECODE_OP( ELEWISECLOSURE,ElewiseOp )
		CASE_CLOSURECODE_OP( SLICECLOSURE,SliceOp )
		CASE_CLOSURECODE_OP( MAXINDEXCLOSURE,MaxIndexOp )
		CASE_CLOSURECODE_OP( RELUFORWARDCLOSURE,ReluForwardOp )
		CASE_CLOSURECODE_OP( ACTIVATIONBACKWARDCLOSURE,ActivationBackwardOp )
		CASE_CLOSURECODE_OP( SELECTCLOSURE,SelectOp )
		CASE_CLOSURECODE_OP( RELUBACKWARDCLOSURE,ReluBackwardOp )
		CASE_CLOSURECODE_OP( TANHBACKWARDCLOSURE,TanhBackwardOp )
		CASE_CLOSURECODE_OP( SYNCWITHPSCLOSURE,SyncWithPSOp )
		CASE_CLOSURECODE_OP( RANDBERNOULLICLOSURE,RandBernoulliOp )
		CASE_CLOSURECODE_OP( CONVBACKWARDDATACLOSURE,ConvBackwardDataOp )
		CASE_CLOSURECODE_OP( FILLCLOSURE,FillOp )
		CASE_CLOSURECODE_OP( LRNBACKWARDCLOSURE,LRNBackwardOp )
		CASE_CLOSURECODE_OP( SOFTMAXFORWARDCLOSURE,SoftmaxForwardOp )
		CASE_CLOSURECODE_OP( ACTIVATIONFORWARDCLOSURE,ActivationForwardOp )
		CASE_CLOSURECODE_OP( RESHAPECLOSURE,ReshapeOp )
		CASE_CLOSURECODE_OP( POOLINGBACKWARDCLOSURE,PoolingBackwardOp )
		CASE_CLOSURECODE_OP( RANDNCLOSURE,RandnOp )
		CASE_CLOSURECODE_OP( MATMULTCLOSURE,MatMultOp )
		CASE_CLOSURECODE_OP( ARITHMETICCLOSURE,ArithmeticOp )
		CASE_CLOSURECODE_OP( CONVBACKWARDFILTERCLOSURE,ConvBackwardFilterOp )
		CASE_CLOSURECODE_OP( POOLINGFORWARDCLOSURE,PoolingForwardOp )
		CASE_CLOSURECODE_OP( ARITHMETICCONSTCLOSURE,ArithmeticConstOp )
		CASE_CLOSURECODE_OP( SIGMOIDBACKWARDCLOSURE,SigmoidBackwardOp )
		CASE_CLOSURECODE_OP( CONCATCLOSURE,ConcatOp )
		CASE_CLOSURECODE_OP( CONVFORWARDCLOSURE,ConvForwardOp )
		CASE_CLOSURECODE_OP( NORMARITHMETICCLOSURE,NormArithmeticOp )
		CASE_CLOSURECODE_OP( SOFTMAXBACKWARDCLOSURE,SoftmaxBackwardOp )
		CASE_CLOSURECODE_OP( LRNFORWARDCLOSURE,LRNForwardOp )
		CASE_CLOSURECODE_OP( INDEXCLOSURE,IndexOp )
		CASE_CLOSURECODE_OP( CONVBACKWARDBIASCLOSURE,ConvBackwardBiasOp )
		CASE_CLOSURECODE_OP( TANHFORWARDCLOSURE,TanhForwardOp )
	default:
		LOG(FATAL) << "Bad closure code" << opcode ;
		throw 100;
	}
}

} //end namespace minerva
