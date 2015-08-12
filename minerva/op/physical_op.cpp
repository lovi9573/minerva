/*
 * closure.cpp
 *
 *  Created on: Jun 16, 2015
 *      Author: jlovitt
 */

#include <string.h>
#include "op/physical_op.h"
#include "op/compute_fn.h"


namespace minerva {


int ArrayLoaderOp::GetSerializedSize() const {
#ifdef HAS_MPI
	return sizeof(int) + sizeof(int) + closure.count*sizeof(element_t);
#else
	return sizeof(int) + sizeof(element_t*);
#endif
}
int ArrayLoaderOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, ARRAYLOADERCLOSURE, int)
#ifdef HAS_MPI
	SERIALIZE(buffer, offset, closure.count, int)
	memcpy(buffer+offset, closure.data.get(), closure.count*sizeof(element_t));
	offset += closure.count*sizeof(element_t);
#else
	SERIALIZE(buffer, offset, closure.data.get(), element_t*);
#endif
	return offset;
}
std::shared_ptr<ComputeFn> ArrayLoaderOp::DeSerialize(char* buffer,int* bytes) {
	ArrayLoaderOp *op = new ArrayLoaderOp();
	int offset = 0;
#ifdef HAS_MPI
	int count;
	DESERIALIZE(buffer, offset, count, int)
	op->closure.count = count;
	std::shared_ptr<element_t> data(new element_t[count], [](element_t* p) {
	    delete[] p;
	  });
	memcpy(data.get(), buffer+offset, count*sizeof(element_t));
	op->closure.data = data;
	*bytes = offset + count*sizeof(element_t);
#else
	element_t *ptr;
	DESERIALIZE(buffer, offset, ptr, element_t*)
	op->closure.data.reset(ptr);
	*bytes = offset ;
#endif
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
std::shared_ptr<ComputeFn> RandnOp::DeSerialize(char* buffer,int* bytes) {
	RandnOp *op = new RandnOp();
	int offset = 0;
	DESERIALIZE(buffer, offset, op->closure.mu, float)
	DESERIALIZE(buffer, offset, op->closure.var, float)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> RandBernoulliOp::DeSerialize(char* buffer,int* bytes) {
	RandBernoulliOp *op = new RandBernoulliOp();
	int offset = 0;
	DESERIALIZE(buffer, offset, op->closure.p, float)
	*bytes = offset;
	return std::shared_ptr<ComputeFn>(op);
}


int FillOp::GetSerializedSize() const  {
	return sizeof(int) + sizeof(element_t);
}
int FillOp::Serialize(char* buffer) const  {
	int offset = 0;
	SERIALIZE(buffer, offset, FILLCLOSURE, int)
	SERIALIZE(buffer, offset, closure.val, element_t)
	return offset;
}
std::shared_ptr<ComputeFn> FillOp::DeSerialize(char* buffer, int* bytes) {
	FillOp *op = new FillOp();
	int offset = 0;
	DESERIALIZE(buffer, offset, op->closure.val, element_t)
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
std::shared_ptr<ComputeFn> SyncWithPSOp::DeSerialize(char* buffer,int* bytes) {
	SyncWithPSOp *op = new SyncWithPSOp();
	NO_IMPL();
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
std::shared_ptr<ComputeFn> MatMultOp::DeSerialize(char* buffer, int* bytes) {
	MatMultOp *op = new MatMultOp();
	*bytes = 0;
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
std::shared_ptr<ComputeFn> TransOp::DeSerialize(char* buffer, int* bytes) {
	TransOp *op = new TransOp();
	*bytes = 0;
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
std::shared_ptr<ComputeFn> ReshapeOp::DeSerialize(char* buffer, int* bytes) {
	ReshapeOp *op = new ReshapeOp();
	*bytes = 0;
	return std::shared_ptr<ComputeFn>(op);
}


int ReductionOp::GetSerializedSize() const {
	return sizeof(int)+ sizeof(ReductionType)+closure.dims_to_reduce.GetSerializedSize() ;
}
int ReductionOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, REDUCTIONCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.type), int)
	offset += closure.dims_to_reduce.Serialize(buffer+offset);
	return offset;
}
std::shared_ptr<ComputeFn> ReductionOp::DeSerialize(char* buffer,int* bytes) {
	ReductionOp *op = new ReductionOp();
	int offset = 0;
	int b = 0;
	DESERIALIZE_ENUM(buffer, offset, op->closure.type, ReductionType)
	op->closure.dims_to_reduce = Scale::DeSerialize(buffer+offset, &b );
	offset += b;
	*bytes = offset;
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
std::shared_ptr<ComputeFn> MaxIndexOp::DeSerialize(char* buffer, int* bytes) {
	MaxIndexOp *op = new MaxIndexOp();
	int offset = 0;
	DESERIALIZE(buffer, offset, op->closure.dim, int)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> ElewiseOp::DeSerialize(char* buffer, int* bytes) {
	ElewiseOp *op = new ElewiseOp();
	int offset = 0;
	DESERIALIZE_ENUM(buffer, offset, op->closure.type, ElewiseType)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> SigmoidForwardOp::DeSerialize(char* buffer, int* bytes) {
	SigmoidForwardOp *op = new SigmoidForwardOp();
	*bytes = 0;
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
std::shared_ptr<ComputeFn> SigmoidBackwardOp::DeSerialize(char* buffer,int* bytes) {
	SigmoidBackwardOp *op = new SigmoidBackwardOp();
	*bytes = 0;
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
std::shared_ptr<ComputeFn> ReluForwardOp::DeSerialize(char* buffer,int* bytes) {
	ReluForwardOp *op = new ReluForwardOp();
	*bytes = 0;
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
std::shared_ptr<ComputeFn> ReluBackwardOp::DeSerialize(char* buffer,int* bytes) {
	ReluBackwardOp *op = new ReluBackwardOp();
	*bytes = 0;
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
std::shared_ptr<ComputeFn> TanhForwardOp::DeSerialize(char* buffer,int* bytes) {
	TanhForwardOp *op = new TanhForwardOp();
	*bytes = 0;
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
std::shared_ptr<ComputeFn> TanhBackwardOp::DeSerialize(char* buffer,int* bytes) {
	TanhBackwardOp *op = new TanhBackwardOp();
	*bytes = 0;
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
	DESERIALIZE_ENUM(buffer, offset, op->closure.type, ArithmeticType)
	*bytes = offset;
	return std::shared_ptr<ComputeFn>(op);
}

int ArithmeticConstOp::GetSerializedSize() const {
	return sizeof(int)+sizeof(ArithmeticType)+sizeof(element_t)+sizeof(int);
}
int ArithmeticConstOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, ARITHMETICCONSTCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.type), int)
	SERIALIZE(buffer, offset, closure.val, element_t)
	SERIALIZE(buffer, offset, closure.side, int)
	return offset;
}
std::shared_ptr<ComputeFn> ArithmeticConstOp::DeSerialize(char* buffer,int* bytes) {
	ArithmeticConstOp *op = new ArithmeticConstOp();
	int offset = 0;
	DESERIALIZE_ENUM(buffer, offset, op->closure.type, ArithmeticType)
	DESERIALIZE(buffer, offset, op->closure.val, element_t)
	DESERIALIZE(buffer, offset, op->closure.side, int)
	*bytes = offset;
	return std::shared_ptr<ComputeFn>(op);
}

int NormArithmeticOp::GetSerializedSize() const {
	return sizeof(int)+ sizeof(ArithmeticType)+closure.dims_to_replicate.GetSerializedSize() ;
}
int NormArithmeticOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, NORMARITHMETICCLOSURE, int)
	SERIALIZE(buffer, offset, static_cast<int>(closure.type), int)
	offset += closure.dims_to_replicate.Serialize(buffer+offset);
	return offset;
}
std::shared_ptr<ComputeFn> NormArithmeticOp::DeSerialize(char* buffer,int* bytes) {
	NormArithmeticOp *op = new NormArithmeticOp();
	int offset = 0;
	int b = 0;
	DESERIALIZE_ENUM(buffer, offset, op->closure.type, ArithmeticType)
	op->closure.dims_to_replicate = Scale::DeSerialize(buffer+offset, &b );
	offset += b;
	*bytes = offset;
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
std::shared_ptr<ComputeFn> ConvForwardOp::DeSerialize(char* buffer,int* bytes) {
	ConvForwardOp *op = new ConvForwardOp();
	int offset = 0;
	DESERIALIZE(buffer, offset, op->closure.pad_height, int)
	DESERIALIZE(buffer, offset, op->closure.pad_width, int)
	DESERIALIZE(buffer, offset, op->closure.stride_vertical, int)
	DESERIALIZE(buffer, offset, op->closure.stride_horizontal, int)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> ConvBackwardDataOp::DeSerialize(char* buffer,int* bytes) {
	ConvBackwardDataOp *op = new ConvBackwardDataOp();
	int offset = 0;
	DESERIALIZE(buffer, offset, op->closure.pad_height, int)
	DESERIALIZE(buffer, offset, op->closure.pad_width, int)
	DESERIALIZE(buffer, offset, op->closure.stride_vertical, int)
	DESERIALIZE(buffer, offset, op->closure.stride_horizontal, int)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> ConvBackwardFilterOp::DeSerialize(char* buffer,int* bytes) {
	ConvBackwardFilterOp *op = new ConvBackwardFilterOp();
	int offset = 0;
	DESERIALIZE(buffer, offset, op->closure.pad_height, int)
	DESERIALIZE(buffer, offset, op->closure.pad_width, int)
	DESERIALIZE(buffer, offset, op->closure.stride_vertical, int)
	DESERIALIZE(buffer, offset, op->closure.stride_horizontal, int)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> ConvBackwardBiasOp::DeSerialize(char* buffer,int* bytes) {
	ConvBackwardBiasOp *op = new ConvBackwardBiasOp();
	*bytes = 0;
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
std::shared_ptr<ComputeFn> SoftmaxForwardOp::DeSerialize(char* buffer,int* bytes) {
	SoftmaxForwardOp *op = new SoftmaxForwardOp();
	int offset = 0;
	DESERIALIZE_ENUM(buffer, offset, op->closure.algorithm, SoftmaxAlgorithm)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> SoftmaxBackwardOp::DeSerialize(char* buffer,int* bytes) {
	SoftmaxBackwardOp *op = new SoftmaxBackwardOp();
	int offset = 0;
	DESERIALIZE_ENUM(buffer, offset, op->closure.algorithm, SoftmaxAlgorithm)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> ActivationForwardOp::DeSerialize(char* buffer,int* bytes) {
	ActivationForwardOp *op = new ActivationForwardOp();
	int offset = 0;
	DESERIALIZE_ENUM(buffer, offset, op->closure.algorithm, ActivationAlgorithm)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> ActivationBackwardOp::DeSerialize(char* buffer,int* bytes) {
	ActivationBackwardOp *op = new ActivationBackwardOp();
	int offset = 0;
	DESERIALIZE_ENUM(buffer, offset, op->closure.algorithm, ActivationAlgorithm)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> PoolingForwardOp::DeSerialize(char* buffer,int* bytes) {
	PoolingForwardOp *op = new PoolingForwardOp();
	int offset = 0;
	DESERIALIZE_ENUM(buffer, offset, op->closure.algorithm, PoolingInfo::Algorithm)
	DESERIALIZE(buffer, offset, op->closure.height, int)
	DESERIALIZE(buffer, offset, op->closure.width, int)
	DESERIALIZE(buffer, offset, op->closure.stride_vertical, int)
	DESERIALIZE(buffer, offset, op->closure.stride_horizontal, int)
	DESERIALIZE(buffer, offset, op->closure.pad_height, int)
	DESERIALIZE(buffer, offset, op->closure.pad_width, int)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> PoolingBackwardOp::DeSerialize(char* buffer,int* bytes) {
	PoolingBackwardOp *op = new PoolingBackwardOp();
	int offset = 0;
	DESERIALIZE_ENUM(buffer, offset, op->closure.algorithm, PoolingInfo::Algorithm)
	DESERIALIZE(buffer, offset, op->closure.height, int)
	DESERIALIZE(buffer, offset, op->closure.width, int)
	DESERIALIZE(buffer, offset, op->closure.stride_vertical, int)
	DESERIALIZE(buffer, offset, op->closure.stride_horizontal, int)
	DESERIALIZE(buffer, offset, op->closure.pad_height, int)
	DESERIALIZE(buffer, offset, op->closure.pad_width, int)
	*bytes = offset;
	return std::shared_ptr<ComputeFn>(op);
}

int LRNForwardOp::GetSerializedSize() const {
	return 2*sizeof(int)+ 2*sizeof(element_t)+closure.data_shape.GetSerializedSize() ;
}
int LRNForwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, LRNFORWARDCLOSURE, int)
	SERIALIZE(buffer, offset, closure.local_size, int)
	SERIALIZE(buffer, offset, closure.alpha, element_t)
	SERIALIZE(buffer, offset, closure.beta, element_t)
	offset += closure.data_shape.Serialize(buffer+offset);
	return offset;
}
std::shared_ptr<ComputeFn> LRNForwardOp::DeSerialize(char* buffer,int* bytes) {
	LRNForwardOp *op = new LRNForwardOp();
	int offset = 0;
	int b = 0;
	DESERIALIZE(buffer, offset, op->closure.local_size, int)
	DESERIALIZE(buffer, offset, op->closure.alpha, element_t)
	DESERIALIZE(buffer, offset, op->closure.beta, element_t)
	op->closure.data_shape = Scale::DeSerialize(buffer+offset, &b );
	offset += b;
	*bytes = offset;
	return std::shared_ptr<ComputeFn>(op);
}


int LRNBackwardOp::GetSerializedSize() const {
	return 2*sizeof(int)+ 2*sizeof(element_t)+closure.data_shape.GetSerializedSize() ;
}
int LRNBackwardOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, LRNBACKWARDCLOSURE, int)
	SERIALIZE(buffer, offset, closure.local_size, int)
	SERIALIZE(buffer, offset, closure.alpha, element_t)
	SERIALIZE(buffer, offset, closure.beta, element_t)
	offset += closure.data_shape.Serialize(buffer+offset);
	return offset;
}
std::shared_ptr<ComputeFn> LRNBackwardOp::DeSerialize(char* buffer,int* bytes) {
	LRNBackwardOp *op = new LRNBackwardOp();
	int offset = 0;
	int b = 0;
	DESERIALIZE(buffer, offset, op->closure.local_size, int)
	DESERIALIZE(buffer, offset, op->closure.alpha, element_t)
	DESERIALIZE(buffer, offset, op->closure.beta, element_t)
	op->closure.data_shape = Scale::DeSerialize(buffer+offset, &b );
	offset += b;
	*bytes = offset;
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
std::shared_ptr<ComputeFn> ConcatOp::DeSerialize(char* buffer,int* bytes) {
	ConcatOp *op = new ConcatOp();
	int offset = 0;
	DESERIALIZE(buffer, offset, op->closure.catdim, int)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> SliceOp::DeSerialize(char* buffer,int* bytes) {
	SliceOp *op = new SliceOp();
	int offset = 0;
	DESERIALIZE(buffer, offset, op->closure.slice_dim, int)
	DESERIALIZE(buffer, offset, op->closure.st_off, int)
	DESERIALIZE(buffer, offset, op->closure.slice_count, int)
	*bytes = offset;
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
std::shared_ptr<ComputeFn> IndexOp::DeSerialize(char* buffer,int* bytes) {
	IndexOp *op = new IndexOp();
	int offset = 0;
	DESERIALIZE(buffer, offset, op->closure.idx, int)
	*bytes = offset;
	return std::shared_ptr<ComputeFn>(op);
}


int SelectOp::GetSerializedSize() const {
	return (closure.indices.size()+1)*sizeof(int);
}
int SelectOp::Serialize(char* buffer) const {
	int offset = 0;
	SERIALIZE(buffer, offset, SELECTCLOSURE, int)
	SERIALIZE(buffer, offset, closure.indices.size(), size_t)
	for(auto& i: closure.indices){
		SERIALIZE(buffer, offset, i, int)
	}
	return offset;
}
std::shared_ptr<ComputeFn> SelectOp::DeSerialize(char* buffer,int* bytes) {
	SelectOp *op = new SelectOp();
	int offset = 0;
	size_t n;
	int val;
	DESERIALIZE(buffer, offset, n, size_t)
	for(size_t i = 0; i < n ; i++){
		DESERIALIZE(buffer, offset, val, int)
		op->closure.indices.emplace_back(val);
	}
	*bytes = offset;
	return std::shared_ptr<ComputeFn>(op);
}


} //end namespace minerva
