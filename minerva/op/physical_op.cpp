/*
 * closure.cpp
 *
 *  Created on: Jun 16, 2015
 *      Author: jlovitt
 */


#include "op/physical_op.h"
#include "op/compute_fn.h"


namespace minerva {


int ArrayLoaderOp::GetSerializedSize() const {
	//TODO: Serialize the array and send it.
	return 0;
}
int ArrayLoaderOp::Serialize(char* buffer) const {
	//TODO: Serialize the array and send it.
	return 0;
}
std::shared_ptr<ComputeFn> ArrayLoaderOp::DeSerialize(char* buffer, int* offset) {
	ArrayLoaderOp *op = new ArrayLoaderOp();
	//TODO: 5 How much of this buffer do we consume?
	//op->closure.data = ((float*)(buffer+*offset));
	//*offset += sizeof(float);
	return std::shared_ptr<ComputeFn>(op);
}

int RandnOp::GetSerializedSize() const {
	return sizeof(int) + 2*sizeof(float);
}
int RandnOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, RANDNCLOSURE, int)
	SERIALIZE(buffer, offset, closure.mu, float)
	SERIALIZE(buffer, offset, closure.var, float)
	return offset;
}

std::shared_ptr<ComputeFn> RandnOp::DeSerialize(char* buffer, int* offset) {
	RandnOp *op = new RandnOp();
	op->closure.mu = *((float*)(buffer+*offset));
	*offset += sizeof(float);
	op->closure.var = *((float*)(buffer+*offset));
	*offset += sizeof(float);
	return std::shared_ptr<ComputeFn>(op);
}

int RandBernoulliOp::GetSerializedSize() const {
	return sizeof(int) + sizeof(float);
}
int RandBernoulliOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, RANDBERNOULLICLOSURE, int)
	SERIALIZE(buffer, offset, closure.p, float)
	return offset;
}
std::shared_ptr<ComputeFn> RandBernoulliOp::DeSerialize(char* buffer, int* offset) {
	RandBernoulliOp *op = new RandBernoulliOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int FillOp::GetSerializedSize() const  {
	return sizeof(int) + sizeof(float);
}
int FillOp::Serialize(char* buffer) const  {
	int offset = 0;
	SERIALIZE(buffer, offset, FILLCLOSURE, int)
	SERIALIZE(buffer, offset, closure.val, float)
	return offset;
}
std::shared_ptr<ComputeFn> FillOp::DeSerialize(char* buffer, int* bytes) {
	FillOp *op = new FillOp();
	int offset = 0;
	//TODO: 5 How much of this buffer do we consume?
	DESERIALIZE(buffer, offset, op->closure.val, float)
	*bytes = offset;
	return std::shared_ptr<ComputeFn>(op);
}

int SyncWithPSOp::GetSerializedSize() const {
	NO_IMPL();
	return 0;
}
int SyncWithPSOp::Serialize(char* buffer) const {
	NO_IMPL();
	return 0;
}
std::shared_ptr<ComputeFn> SyncWithPSOp::DeSerialize(char* buffer, int* offset) {
	SyncWithPSOp *op = new SyncWithPSOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


int MatMultOp::GetSerializedSize() const {
	return sizeof(int);
}
int MatMultOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, MATMULTCLOSURE, int)
	return offset;
}
std::shared_ptr<ComputeFn> MatMultOp::DeSerialize(char* buffer, int* offset) {
	MatMultOp *op = new MatMultOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int TransOp::GetSerializedSize() const {
	return sizeof(int);
}
int TransOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, TRANSPOSECLOSURE, int)
	return offset;
}
std::shared_ptr<ComputeFn> TransOp::DeSerialize(char* buffer, int* offset) {
	TransOp *op = new TransOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int ReshapeOp::GetSerializedSize() const {
	return sizeof(int);
}
int ReshapeOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, RESHAPECLOSURE, int)
	return offset;
}
std::shared_ptr<ComputeFn> ReshapeOp::DeSerialize(char* buffer, int* offset) {
	ReshapeOp *op = new ReshapeOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


int ReductionOp::GetSerializedSize() const {
	return sizeof(int)+ sizeof(ReductionType)+closure.dims_to_reduce.GetSerializedSize() ;
}
int ReductionOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, REDUCTIONCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.type), int)
	offset += closure.dims_to_reduce.Serialize(buffer);
	return offset;
}
std::shared_ptr<ComputeFn> ReductionOp::DeSerialize(char* buffer, int* offset) {
	ReductionOp *op = new ReductionOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int MaxIndexOp::GetSerializedSize() const {
	return 2*sizeof(int);
}
int MaxIndexOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, MAXINDEXCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.dim), int)
	return offset;
}
std::shared_ptr<ComputeFn> MaxIndexOp::DeSerialize(char* buffer, int* offset) {
	MaxIndexOp *op = new MaxIndexOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int ElewiseOp::GetSerializedSize() const {
	return sizeof(int)+ sizeof(ElewiseType);
}
int ElewiseOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, ELEWISECLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.type), int)
	return offset;
}
std::shared_ptr<ComputeFn> ElewiseOp::DeSerialize(char* buffer, int* offset) {
	ElewiseOp *op = new ElewiseOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


int SigmoidForwardOp::GetSerializedSize() const {
	return sizeof(int);
}
int SigmoidForwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, SIGMOIDFORWARDCLOSURE, int)
	return offset;
}
std::shared_ptr<ComputeFn> SigmoidForwardOp::DeSerialize(char* buffer, int* offset) {
	SigmoidForwardOp *op = new SigmoidForwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}



int SigmoidBackwardOp::GetSerializedSize() const {
	return sizeof(int);
}
int SigmoidBackwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, SIGMOIDBACKWARDCLOSURE, int)
	return offset;
}
std::shared_ptr<ComputeFn> SigmoidBackwardOp::DeSerialize(char* buffer, int* offset) {
	SigmoidBackwardOp *op = new SigmoidBackwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


int ReluForwardOp::GetSerializedSize() const {
	return sizeof(int);
}
int ReluForwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, RELUFORWARDCLOSURE, int)
	return offset;
}
std::shared_ptr<ComputeFn> ReluForwardOp::DeSerialize(char* buffer, int* offset) {
	ReluForwardOp *op = new ReluForwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int ReluBackwardOp::GetSerializedSize() const {
	return sizeof(int);
}
int ReluBackwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, RELUBACKWARDCLOSURE, int)
	return offset;
}
std::shared_ptr<ComputeFn> ReluBackwardOp::DeSerialize(char* buffer, int* offset) {
	ReluBackwardOp *op = new ReluBackwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


int TanhForwardOp::GetSerializedSize() const {
	return sizeof(int);
}
int TanhForwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, TANHFORWARDCLOSURE, int)
	return offset;
}
std::shared_ptr<ComputeFn> TanhForwardOp::DeSerialize(char* buffer, int* offset) {
	TanhForwardOp *op = new TanhForwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int TanhBackwardOp::GetSerializedSize() const {
	return sizeof(int);
}
int TanhBackwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, TANHBACKWARDCLOSURE, int)
	return offset;
}
std::shared_ptr<ComputeFn> TanhBackwardOp::DeSerialize(char* buffer, int* offset) {
	TanhBackwardOp *op = new TanhBackwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int ArithmeticOp::GetSerializedSize() const {
	return sizeof(int)+sizeof(ArithmeticType);
}
int ArithmeticOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, ARITHMETICCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.type), int)
	return offset;
}
std::shared_ptr<ComputeFn> ArithmeticOp::DeSerialize(char* buffer, int* bytes) {
	ArithmeticOp *op = new ArithmeticOp();
	int offset = 0;
	//TODO: 5 How much of this buffer do we consume?
	DESERIALIZE_ENUM(buffer, offset, op->closure.type, ArithmeticType)
	*bytes = offset;
	return std::shared_ptr<ComputeFn>(op);
}

int ArithmeticConstOp::GetSerializedSize() const {
	return sizeof(int)+sizeof(ArithmeticType)+sizeof(float)+sizeof(int);
}
int ArithmeticConstOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, ARITHMETICCONSTCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.type), int)
	SERIALIZE(buffer, offset, closure.val, float)
	SERIALIZE(buffer, offset, closure.side, int)
	return offset;
}
std::shared_ptr<ComputeFn> ArithmeticConstOp::DeSerialize(char* buffer, int* offset) {
	ArithmeticConstOp *op = new ArithmeticConstOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int NormArithmeticOp::GetSerializedSize() const {
	return sizeof(int)+ sizeof(ArithmeticType)+closure.dims_to_replicate.GetSerializedSize() ;
}
int NormArithmeticOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, NORMARITHMETICCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.type), int)
	offset += closure.dims_to_replicate.Serialize(buffer);
	return offset;
}
std::shared_ptr<ComputeFn> NormArithmeticOp::DeSerialize(char* buffer, int* offset) {
	NormArithmeticOp *op = new NormArithmeticOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int ConvForwardOp::GetSerializedSize() const {
	return 5*sizeof(int);
}
int ConvForwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, CONVFORWARDCLOSURE, int)
	SERIALIZE(buffer, offset, closure.pad_height, int)
	SERIALIZE(buffer, offset, closure.pad_width, int)
	SERIALIZE(buffer, offset, closure.stride_vertical, int)
	SERIALIZE(buffer, offset, closure.stride_horizontal, int)
	return offset;
}
std::shared_ptr<ComputeFn> ConvForwardOp::DeSerialize(char* buffer, int* offset) {
	ConvForwardOp *op = new ConvForwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int ConvBackwardDataOp::GetSerializedSize() const {
	return 5*sizeof(int);
}
int ConvBackwardDataOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, CONVBACKWARDDATACLOSURE, int)
	SERIALIZE(buffer, offset, closure.pad_height, int)
	SERIALIZE(buffer, offset, closure.pad_width, int)
	SERIALIZE(buffer, offset, closure.stride_vertical, int)
	SERIALIZE(buffer, offset, closure.stride_horizontal, int)
	return offset;
}
std::shared_ptr<ComputeFn> ConvBackwardDataOp::DeSerialize(char* buffer, int* offset) {
	ConvBackwardDataOp *op = new ConvBackwardDataOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


int ConvBackwardFilterOp::GetSerializedSize() const {
	return 5*sizeof(int);
}
int ConvBackwardFilterOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, CONVBACKWARDFILTERCLOSURE, int)
	SERIALIZE(buffer, offset, closure.pad_height, int)
	SERIALIZE(buffer, offset, closure.pad_width, int)
	SERIALIZE(buffer, offset, closure.stride_vertical, int)
	SERIALIZE(buffer, offset, closure.stride_horizontal, int)
	return offset;
}
std::shared_ptr<ComputeFn> ConvBackwardFilterOp::DeSerialize(char* buffer, int* offset) {
	ConvBackwardFilterOp *op = new ConvBackwardFilterOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


int ConvBackwardBiasOp::GetSerializedSize() const {
	return sizeof(int);
}
int ConvBackwardBiasOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, CONVBACKWARDBIASCLOSURE, int)
	return offset;
}
std::shared_ptr<ComputeFn> ConvBackwardBiasOp::DeSerialize(char* buffer, int* offset) {
	ConvBackwardBiasOp *op = new ConvBackwardBiasOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int SoftmaxForwardOp::GetSerializedSize() const {
	return sizeof(int)+sizeof(SoftmaxAlgorithm);
}
int SoftmaxForwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, SOFTMAXFORWARDCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.algorithm), int)
	return offset;
}
std::shared_ptr<ComputeFn> SoftmaxForwardOp::DeSerialize(char* buffer, int* offset) {
	SoftmaxForwardOp *op = new SoftmaxForwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


int SoftmaxBackwardOp::GetSerializedSize() const {
	return sizeof(int)+sizeof(SoftmaxAlgorithm);
}
int SoftmaxBackwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, SOFTMAXBACKWARDCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.algorithm), int)
	return offset;
}
std::shared_ptr<ComputeFn> SoftmaxBackwardOp::DeSerialize(char* buffer, int* offset) {
	SoftmaxBackwardOp *op = new SoftmaxBackwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int ActivationForwardOp::GetSerializedSize() const {
	return sizeof(int)+sizeof(ActivationAlgorithm);
}
int ActivationForwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, ACTIVATIONFORWARDCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.algorithm), int)
	return offset;
}
std::shared_ptr<ComputeFn> ActivationForwardOp::DeSerialize(char* buffer, int* offset) {
	ActivationForwardOp *op = new ActivationForwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


int ActivationBackwardOp::GetSerializedSize() const {
	return sizeof(int)+sizeof(ActivationAlgorithm);
}
int ActivationBackwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, ACTIVATIONBACKWARDCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.algorithm), int)
	return offset;
}
std::shared_ptr<ComputeFn> ActivationBackwardOp::DeSerialize(char* buffer, int* offset) {
	ActivationBackwardOp *op = new ActivationBackwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int PoolingForwardOp::GetSerializedSize() const {
	return 7*sizeof(int)+sizeof(PoolingInfo::Algorithm);
}
int PoolingForwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, POOLINGFORWARDCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.algorithm), int)
	SERIALIZE(buffer, offset, closure.height, int)
	SERIALIZE(buffer, offset, closure.width, int)
	SERIALIZE(buffer, offset, closure.stride_vertical, int)
	SERIALIZE(buffer, offset, closure.stride_horizontal, int)
	SERIALIZE(buffer, offset, closure.pad_height, int)
	SERIALIZE(buffer, offset, closure.pad_width, int)
	return offset;
}
std::shared_ptr<ComputeFn> PoolingForwardOp::DeSerialize(char* buffer, int* offset) {
	PoolingForwardOp *op = new PoolingForwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int PoolingBackwardOp::GetSerializedSize() const {
	return 7*sizeof(int)+sizeof(PoolingInfo::Algorithm);
}
int PoolingBackwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, POOLINGBACKWARDCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.algorithm), int)
	SERIALIZE(buffer, offset, closure.height, int)
	SERIALIZE(buffer, offset, closure.width, int)
	SERIALIZE(buffer, offset, closure.stride_vertical, int)
	SERIALIZE(buffer, offset, closure.stride_horizontal, int)
	SERIALIZE(buffer, offset, closure.pad_height, int)
	SERIALIZE(buffer, offset, closure.pad_width, int)
	return offset;
}
std::shared_ptr<ComputeFn> PoolingBackwardOp::DeSerialize(char* buffer, int* offset) {
	PoolingBackwardOp *op = new PoolingBackwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int LRNForwardOp::GetSerializedSize() const {
	return 2*sizeof(int)+ 2*sizeof(float)+closure.data_shape.GetSerializedSize() ;
}
int LRNForwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, LRNFORWARDCLOSURE, int)
	SERIALIZE(buffer, offset, closure.local_size, int)
	SERIALIZE(buffer, offset, closure.alpha, float)
	SERIALIZE(buffer, offset, closure.beta, float)
	offset += closure.data_shape.Serialize(buffer);
	return offset;
}
std::shared_ptr<ComputeFn> LRNForwardOp::DeSerialize(char* buffer, int* offset) {
	LRNForwardOp *op = new LRNForwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


int LRNBackwardOp::GetSerializedSize() const {
	return 2*sizeof(int)+ 2*sizeof(float)+closure.data_shape.GetSerializedSize() ;
}
int LRNBackwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, LRNBACKWARDCLOSURE, int)
	SERIALIZE(buffer, offset, closure.local_size, int)
	SERIALIZE(buffer, offset, closure.alpha, float)
	SERIALIZE(buffer, offset, closure.beta, float)
	offset += closure.data_shape.Serialize(buffer);
	return offset;
}
std::shared_ptr<ComputeFn> LRNBackwardOp::DeSerialize(char* buffer, int* offset) {
	LRNBackwardOp *op = new LRNBackwardOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int ConcatOp::GetSerializedSize() const {
	return 2*sizeof(int);
}
int ConcatOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, CONCATCLOSURE, int)
	SERIALIZE(buffer, offset, closure.catdim, int)
	return offset;
}
std::shared_ptr<ComputeFn> ConcatOp::DeSerialize(char* buffer, int* offset) {
	ConcatOp *op = new ConcatOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int SliceOp::GetSerializedSize() const {
	return 4*sizeof(int);
}
int SliceOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, SLICECLOSURE, int)
	SERIALIZE(buffer, offset, closure.slice_dim, int)
	SERIALIZE(buffer, offset, closure.st_off, int)
	SERIALIZE(buffer, offset, closure.slice_count, int)
	return offset;
}
std::shared_ptr<ComputeFn> SliceOp::DeSerialize(char* buffer, int* offset) {
	SliceOp *op = new SliceOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}

int IndexOp::GetSerializedSize() const {
	return 2*sizeof(int);
}
int IndexOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, INDEXCLOSURE, int)
	SERIALIZE(buffer, offset, closure.idx, int)
	return offset;
}
std::shared_ptr<ComputeFn> IndexOp::DeSerialize(char* buffer, int* offset) {
	IndexOp *op = new IndexOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


int SelectOp::GetSerializedSize() const {
	return (closure.indices.size()+1)*sizeof(int);
}
int SelectOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, SELECTCLOSURE, int)
	for(auto& i: closure.indices){
		SERIALIZE(buffer, offset, i, int)
	}
	return offset;
}
std::shared_ptr<ComputeFn> SelectOp::DeSerialize(char* buffer, int* offset) {
	SelectOp *op = new SelectOp();
	//TODO: 5 How much of this buffer do we consume?
	return std::shared_ptr<ComputeFn>(op);
}


} //end namespace minerva
