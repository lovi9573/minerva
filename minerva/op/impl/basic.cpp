#include "basic.h"
#include "op/closure.h"
#include <cmath>
#include <dmlc/logging.h>
#include <chrono>
#include <algorithm>
#include <random>
#include <cstring>

using namespace std;

#ifdef HAS_PS
#include "op/impl/ps.h"
#endif

#ifdef HAS_CBLAS
#include <cblas.h>
#endif

namespace minerva {
namespace basic {

void Arithmetic(const DataList& inputs, const DataList& outputs, ArithmeticClosure& closure) {
  CHECK_EQ(inputs.size(), 2) << "(arithmetic) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(arithmetic) #outputs is wrong!";
  element_t* left_data = inputs[0].data_;
  element_t* right_data = inputs[1].data_;
  element_t* res_data = outputs[0].data_;
  int length = outputs[0].size_.Prod();
  switch (closure.type) {
    case ArithmeticType::kAdd:
      for (int i = 0; i < length; ++i) {
        res_data[i] = left_data[i] + right_data[i];
      }
      break;
    case ArithmeticType::kSub:
      for (int i = 0; i < length; ++i) {
        res_data[i] = left_data[i] - right_data[i];
      }
      break;
    case ArithmeticType::kMult:
      for (int i = 0; i < length; ++i) {
        res_data[i] = left_data[i] * right_data[i];
      }
      break;
    case ArithmeticType::kDiv:
      for (int i = 0; i < length; ++i) {
        res_data[i] = left_data[i] / right_data[i];
      }
      break;
  }
}

void ArithmeticConst(const DataList& inputs, const DataList& outputs, ArithmeticConstClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "(arithmetic const) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(arithmetic const) #outputs is wrong!";
  element_t val = closure.val;
  element_t* in_data = inputs[0].data_;
  element_t* res_data = outputs[0].data_;
  int length = outputs[0].size_.Prod();
  switch (closure.type) {
    case ArithmeticType::kAdd:
      if (closure.side == 0) {  // const on left
        for (int i = 0; i < length; ++i) {
          res_data[i] = val + in_data[i];
        }
      } else {  // const on right
        for (int i = 0; i < length; ++i) {
          res_data[i] = in_data[i] + val;
        }
      }
      break;
    case ArithmeticType::kSub:
      if (closure.side == 0) {  // const on left
        for (int i = 0; i < length; ++i) {
          res_data[i] = val - in_data[i];
        }
      } else {  // const on right
        for (int i = 0; i < length; ++i) {
          res_data[i] = in_data[i] - val;
        }
      }
      break;
    case ArithmeticType::kMult:
      if (closure.side == 0) {  // const on left
        for (int i = 0; i < length; ++i) {
          res_data[i] = val * in_data[i];
        }
      } else {  // const on right
        for (int i = 0; i < length; ++i) {
          res_data[i] = in_data[i] * val;
        }
      }
      break;
    case ArithmeticType::kDiv:
      if (closure.side == 0) {  // const on left
        for (int i = 0; i < length; ++i) {
          res_data[i] = val / in_data[i];
        }
      } else {  // const on right
        for (int i = 0; i < length; ++i) {
          res_data[i] = in_data[i] / val;
        }
      }
      break;
  }
}


void ThresholdNorm(const DataList& inputs, const DataList& outputs, ThresholdNormClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "(elewise) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(elewise) #outputs is wrong!";
  element_t* in_data = inputs[0].data_;
  element_t* res_data = outputs[0].data_;
  int length = outputs[0].size_.Prod();
  for (int i = 0; i < length; ++i) {
	//Suppose threshold can't be zero
	if(std::abs(in_data[i]) > closure.threshold)
		res_data[i] = std::abs(in_data[i]) / in_data[i] * closure.threshold;
	else
		res_data[i] = in_data[i];
  }
}

void Elewise(const DataList& inputs, const DataList& outputs, ElewiseClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "(elewise) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(elewise) #outputs is wrong!";
  element_t* in_data = inputs[0].data_;
  element_t* res_data = outputs[0].data_;
  int length = outputs[0].size_.Prod();
  switch (closure.type) {
    case ElewiseType::kExp:
      for (int i = 0; i < length; ++i) {
        res_data[i] = exp(in_data[i]);
      }
      break;
    case ElewiseType::kLn:
      for (int i = 0; i < length; ++i) {
        res_data[i] = log(in_data[i]);
      }
      break;
    case ElewiseType::kNegative:
      for (int i = 0; i < length; ++i) {
        res_data[i] = -in_data[i];
      }
      break;
  }
}

void MatMult(const DataList& inputs, const DataList& outputs, MatMultClosure& closure) {
  CHECK_EQ(inputs.size(), 2) << "(matmult) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(matmult) #outputs is wrong!";
  element_t* left_data = inputs[0].data_;
  element_t* right_data = inputs[1].data_;
  element_t* res_data = outputs[0].data_;
  int m = outputs[0].size_[0];
  int n = outputs[0].size_[1];
  int o = inputs[0].size_[1];
  // ATTENTION: the data is column major !!
#ifdef HAS_CBLAS
  memset(res_data, 0, sizeof(element_t) * m * n);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, o, 1.0, left_data, m, right_data, o, 0.0, res_data, m);
#else
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      res_data[i + j * m] = 0;
      for (int k = 0; k < o; ++k) {
        res_data[i + j * m] += left_data[i + k * m] * right_data[k + j * o];
      }
    }
  }
#endif
}

void Transpose(const DataList& inputs, const DataList& outputs, TransposeClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "(transpose) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(transpose) #outputs is wrong!";
  element_t* in_data = inputs[0].data_;
  element_t* res_data = outputs[0].data_;
  int m = outputs[0].size_[0];
  int n = outputs[0].size_[1];
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      res_data[i + j * m] = in_data[j + i * n];
    }
  }
}

void Reduction(const DataList& inputs, const DataList& outputs, ReductionClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "(reduction) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(reduction) #outputs is wrong!";
  element_t* in_data = inputs[0].data_;
  element_t* res_data = outputs[0].data_;
  auto in_max = inputs[0].size_;
  auto in_range = ScaleRange::MakeRangeFromOrigin(in_max);
  auto res_max = outputs[0].size_;
  auto res_range = ScaleRange::MakeRangeFromOrigin(res_max);
  auto accumulator = Scale::Origin(in_max.NumDims());
  do {
    auto cur = accumulator;
    element_t tmp = in_data[in_range.Flatten(cur)];
    while (cur.IncrDimensions(in_max, closure.dims_to_reduce)) {
      element_t tmp2 = in_data[in_range.Flatten(cur)];
      switch (closure.type) {
        case ReductionType::kSum:
          tmp += tmp2;
          break;
        case ReductionType::kMax:
          if (tmp < tmp2) {
            tmp = tmp2;
          }
          break;
      }
    }
    res_data[res_range.Flatten(accumulator)] = tmp;
  } while (accumulator.IncrWithDimensionsFixed(res_max, closure.dims_to_reduce));
}

void SoftmaxForward(const DataList& inputs, const DataList& outputs, SoftmaxForwardClosure& closure) {
  //TODO: Currently CPU only support kInstance softmax
  CHECK_EQ(inputs.size(), 1) << "(reduction) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(reduction) #outputs is wrong!";
  element_t* in_data = inputs[0].data_;
  element_t* res_data = outputs[0].data_;
  auto in_max = inputs[0].size_;
  auto in_range = ScaleRange::MakeRangeFromOrigin(in_max);
  auto res_max = outputs[0].size_;
  auto res_range = ScaleRange::MakeRangeFromOrigin(res_max);
  auto accumulator = Scale::Origin(in_max.NumDims());

  //normalize according to batch dimension
  std::vector<int> batchdim(1,0);
  auto dim_to_norm = Scale(batchdim);
  //sub the max to prevent numerical problem
  do {
    auto cur = accumulator;
    element_t tmp = in_data[in_range.Flatten(cur)];
    //get max
    while (cur.IncrDimensions(in_max, dim_to_norm)) {
      element_t tmp2 = in_data[in_range.Flatten(cur)];
      if (tmp < tmp2) {
        tmp = tmp2;
      }
    }
    //exp(x - max), also sum the result
    cur = accumulator;
	if (cur != in_max) {
		element_t sum_exp = 0;
		do {
		  res_data[in_range.Flatten(cur)] = expf(in_data[in_range.Flatten(cur)] - tmp);
		  sum_exp += res_data[in_range.Flatten(cur)];
		}while (cur.IncrDimensions(in_max, dim_to_norm));
		//devide the sum
		cur = accumulator;
		do {
		  res_data[in_range.Flatten(cur)] /= sum_exp;
		}while (cur.IncrDimensions(in_max, dim_to_norm));
	}
  } while (accumulator.IncrWithDimensionsFixed(res_max, dim_to_norm));
}

void ArrayLoader(const DataList& outputs, ArrayLoaderClosure& closure) {
  CHECK_EQ(outputs.size(), 1) << "(array loader) #outputs wrong";
  CHECK(closure.data) << "probably already executed";
  memcpy(outputs[0].data_, closure.data.get(), outputs[0].size_.Prod() * sizeof(element_t));
  closure.data.reset();
}

void Randn(const DataList& output, RandnClosure& closure) {
  CHECK_EQ(output.size(), 1) << "wrong number of randn output";
  int length = output[0].size_.Prod();
  element_t* data = output[0].data_;
  default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
  normal_distribution<float> distribution(closure.mu, closure.var);
  for (int i = 0; i < length; ++i) {
    data[i] = distribution(generator);
  }
}

void RandBernoulli(const DataList& outputs, RandBernoulliClosure& closure) {
  CHECK_EQ(outputs.size(), 1) << "(bernoulli) #outputs wrong";
  int length = outputs[0].size_.Prod();
  element_t* data = outputs[0].data_;
  default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
  bernoulli_distribution distribution(closure.p);
  for (int i = 0; i < length; ++i) {
    data[i] = distribution(generator);
  }
}

void Fill(const DataList& output, FillClosure& closure) {
  CHECK_EQ(output.size(), 1) << "wrong number of fill constant output";
  int length = output[0].size_.Prod();
  element_t* data = output[0].data_;
  for (int i = 0; i < length; ++i) {
    data[i] = closure.val;
  }
}

void SyncWithPS(const DataList& inputs, const DataList& outputs, SyncWithPSClosure& closure) {
  CHECK_EQ(outputs.size(), 1);
#ifdef HAS_PS
  if (inputs.empty())
  {
    PushGradAndPullWeight(nullptr, outputs[0].data_, outputs[0].size_.Prod(), closure.layer_name);
  }
  else
  {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size_.Prod(), outputs[0].size_.Prod()) << "Pushed and pulled matrix must be of same dim";
    PushGradAndPullWeight(inputs[0].data_, outputs[0].data_, inputs[0].size_.Prod(), closure.layer_name);
  }
#else
  LOG(FATAL) << "HAS_PS is not enabled when you compile minerva, please enable it";
#endif
}

void NormArithmetic(const DataList& inputs, const DataList& outputs, NormArithmeticClosure& closure) {
  CHECK_EQ(inputs.size(), 2) << "NormArithmetic kernel wrong #input";
  CHECK_EQ(outputs.size(), 1) << "NormArithmetic kernel wrong #output";
  // Normalizee is the chunk with full size, normalizer is the chunk with reduced dimensions
  auto normalizee_size = inputs[0].size_;
  auto normalizer_size = inputs[1].size_;
  auto normalizee_range = ScaleRange::MakeRangeFromOrigin(normalizee_size);
  auto normalizer_range = ScaleRange::MakeRangeFromOrigin(normalizer_size);
  auto normalizee_data = inputs[0].data_;
  auto normalizer_data = inputs[1].data_;
  auto res_data = outputs[0].data_;
  // Memory copy
  memcpy(res_data, normalizee_data, normalizee_size.Prod() * sizeof(element_t));
  // Reuse of single element per iteration
  size_t single_iteration_size = 1;
  for (size_t i = 0; i < normalizee_size.NumDims(); ++i) {
    if (!closure.dims_to_replicate.Contains(i)) {
      break;
    }
    single_iteration_size *= normalizee_size[i];
  }
  auto iterator = Scale::Origin(normalizee_size.NumDims());
  bool no_end = true;
  while (no_end) {
    auto iterator_normalizer = iterator;
    for (auto i: closure.dims_to_replicate) {
      iterator_normalizer[i] = 0;
    }
    element_t cur = normalizer_data[normalizer_range.Flatten(iterator_normalizer)];
    size_t flatten = normalizee_range.Flatten(iterator);
    for (size_t i = 0; i < single_iteration_size; ++i) {
      switch (closure.type) {
        case ArithmeticType::kAdd:
          res_data[flatten + i] += cur;
          break;
        case ArithmeticType::kSub:
          res_data[flatten + i] -= cur;
          break;
        case ArithmeticType::kMult:
          res_data[flatten + i] *= cur;
          break;
        case ArithmeticType::kDiv:
          res_data[flatten + i] /= cur;
          break;
      }
      no_end = iterator.IncrOne(normalizee_size);
    }
  }
}

void MaxIndex(const DataList& inputs, const DataList& outputs, MaxIndexClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "basic::MaxIndex #input wrong";
  CHECK_EQ(outputs.size(), 1) << "basic::MaxIndex #output wrong";
  element_t* in_data = inputs[0].data_;
  element_t* res_data = outputs[0].data_;
  auto in_max = inputs[0].size_;
  auto in_range = ScaleRange::MakeRangeFromOrigin(in_max);
  auto res_max = outputs[0].size_;
  auto res_range = ScaleRange::MakeRangeFromOrigin(res_max);
  Scale dims{closure.dim};
  // Interval and strike for a single iteration
  int interval = 1;
  for (int i = 0; i < closure.dim; ++i) {
    interval *= inputs[0].size_[i];
  }
  int strike = inputs[0].size_[closure.dim];
  auto iterator = Scale::Origin(in_max.NumDims());
  do {
    size_t offset = in_range.Flatten(iterator);
    element_t cur_max = in_data[offset];
    int index = 0;
    for (int i = 0; i < strike; ++i) {
      if (cur_max < in_data[offset + i * interval]) {
        cur_max = in_data[offset + i * interval];
        index = i;
      }
    }
    res_data[res_range.Flatten(iterator)] = index;
  } while (iterator.IncrWithDimensionsFixed(res_max, dims));
}

void Reshape(const DataList& inputs, const DataList& outputs, ReshapeClosure&) {
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  memcpy(outputs[0].data_, inputs[0].data_, inputs[0].size_.Prod() * sizeof(element_t));
}

void SigmoidForward(const DataList& inputs, const DataList& outputs, SigmoidForwardClosure&) {
  CHECK_EQ(inputs.size(), 1) << "sigmoid forward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "sigmoid forward #outputs wrong";

  element_t* input_data = inputs[0].data_;
  element_t* output_data = outputs[0].data_;

  size_t numbers = inputs[0].size_.Prod();

  for (size_t i = 0; i < numbers; i++) {
    output_data[i] = 1.0 / (1.0 + expf(-input_data[i]));
  }
}

void ReluForward(const DataList& inputs, const DataList& outputs, ReluForwardClosure&) {
  CHECK_EQ(inputs.size(), 1) << "relu forward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "relu forward #outputs wrong";
  CHECK_EQ(outputs[0].size_.Prod(), inputs[0].size_.Prod()) << "top and bottom sizes aren't the same.\n";

  element_t* input_data = inputs[0].data_;
  element_t* output_data = outputs[0].data_;

  size_t numbers = inputs[0].size_.Prod();

  for (size_t i = 0; i < numbers; i++) {
    if( input_data[i] > 0.0) {
    	output_data[i] = input_data[i];
    }else{
    	output_data[i] = 0.0;
    }
  }
}

void ReluBackward(const DataList& inputs, const DataList& outputs, ReluBackwardClosure&) {
	  CHECK_EQ(inputs.size(), 3) << "relu backward #inputs wrong";
	  CHECK_EQ(outputs.size(), 1) << "relu backward #outputs wrong";

	  element_t* diff = inputs[0].data_;
	  element_t* top = inputs[1].data_;
	  //element_t* bottom = inputs[2].data_;
	  element_t* bottom_diff = outputs[0].data_;

	  size_t numbers = inputs[0].size_.Prod();

	  for(size_t i = 0; i < numbers; i++){
		  if(top[i] > 0.0){
			  bottom_diff[i] = diff[i];
		  }else{
			  bottom_diff[i] = 0.0;
		  }
	  }
}

void TanhForward(const DataList& inputs, const DataList& outputs, TanhForwardClosure&) {
  CHECK_EQ(inputs.size(), 1) << "tanh forward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "tanh forward #outputs wrong";

  element_t* input_data = inputs[0].data_;
  element_t* output_data = outputs[0].data_;

  size_t numbers = inputs[0].size_.Prod();

  for (size_t i = 0; i < numbers; i++) {
    output_data[i] = tanhf(input_data[i]);
  }
}

void ActivationForward(const DataList& inputs, const DataList& outputs, ActivationForwardClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "(activation forward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(activation forward) #outputs wrong";
  switch (closure.algorithm) {
    case ActivationAlgorithm::kSigmoid: {
      SigmoidForwardClosure c;
      SigmoidForward(inputs, outputs, c);
      break;
    }
    case ActivationAlgorithm::kRelu: {
      ReluForwardClosure c;
      ReluForward(inputs, outputs, c);
      break;
    }
    case ActivationAlgorithm::kTanh: {
      TanhForwardClosure c;
      TanhForward(inputs, outputs, c);
      break;
    }
    default:
      LOG(FATAL) << "activation algorithm not supported";
  }
}

/*
 * NOTE: This implementation is not designed to be fast.  If you want that, use a GPU.  It is meant to provide feature completeness and the ability to develop on a GPU-less machine.
 */
void ConvForward(const DataList& inputs, const DataList& outputs, ConvForwardClosure& closure){
	CHECK_EQ(inputs.size(), 3) << " Convolution #inputs are wrong.\n";
	CHECK_EQ(outputs.size(), 1) << " Convolution #outputs are wrong.\n";

	element_t* bottom = inputs[0].data_;
	Scale bottom_size = inputs[0].size_;
	int bottom_column_stride = bottom_size[0];
	int bottom_channel_stride = bottom_size[1]*bottom_size[0];
	int bottom_image_stride = bottom_size[2]*bottom_size[1]*bottom_size[0];

	element_t* filters = inputs[1].data_;
	Scale filtersize = inputs[1].size_;
	int filter_column_stride = filtersize[0];
	int filter_channel_stride = filtersize[1]*filtersize[0];
	int filter_element_stride = filtersize[2]*filtersize[1]*filtersize[0];

	element_t* bias = inputs[2].data_;

	element_t* top = outputs[0].data_;
	Scale top_size = outputs[0].size_;
	//int top_column_stride = top_size[0];
	//int top_channel_stride = top_size[1]*top_size[0];
	//int top_image_stride = top_size[2]*top_size[1]*top_size[0];

/*
	for(int i = 0; i < top_size.Prod(); i++){
		top[i] = 0.0f;
	}
*/
	//printf("bottom #channels: %d\n",bottom_size[2]);

    int pad_height = closure.pad_height;
    int pad_width = closure.pad_width;
    int stride_vertical = closure.stride_vertical;
    int stride_horizontal = closure.stride_horizontal;

    int outindex = -1  ;

	for(int element = 0 ; element < bottom_size[3]; element++){
		for(int filter = 0; filter < filtersize[3]; filter++){
			int filter_offset = filter_element_stride*filter;
			for(int y = -pad_height; y <= bottom_size[1]-filtersize[1]+pad_height; y = y + stride_vertical){
				for(int x = -pad_width; x <= bottom_size[0]-filtersize[0]+pad_width; x += stride_horizontal){
					outindex++;
					top[outindex] = 0.0f;
					for(int channel =0; channel < bottom_size[2]; channel++){
						for(int filter_y = 0; filter_y < filtersize[1]; filter_y++){
							for(int filter_x = 0; filter_x < filtersize[0]; filter_x++){
								//printf("\tx+filter_x: %d, y+filter_y: %d\n",x+filter_x, y+filter_y);
								if(x+filter_x >= 0  && y+filter_y >= 0 && x+filter_x < bottom_size[0] && y+filter_y < bottom_size[1]){
									//printf("IN %d\n",outindex);
									int inindex = (x+filter_x)+bottom_column_stride*(y+filter_y)+bottom_channel_stride*channel+bottom_image_stride*element ;
									int filter_index = (filtersize[0] -1 -filter_x)+filter_column_stride*(filtersize[1] -1 -filter_y)+filter_channel_stride*channel+filter_offset;
									top[outindex ] += bottom[inindex] * filters[filter_index] ;
									/*
									if(outindex == 4){
										printf("filter: (%d,%d) %f * bottom: (%d,%d) %f\n",
												x +filter_x, y + filter_y,filters[filter_index],
												filter_x, filter_y,bottom[inindex]);
									}
									*/
								}
							}
						}
					}//End full filter (filter_x*filter_y*channel)
					top[outindex]  += bias[filter];
				}
			}
		}
	}
}

/*
 * NOTE: This implementation is not designed to be fast.  If you want that, use a GPU.  It is meant to provide feature completeness and the ability to develop on a GPU-less machine.
 */
void ConvBackwardData(const DataList& inputs, const DataList& outputs, ConvBackwardDataClosure& closure){
	CHECK_EQ(inputs.size(), 2) << " Convolution #inputs are wrong.\n";
	CHECK_EQ(outputs.size(), 1) << " Convolution #outputs are wrong.\n";

	element_t* bottom_diff = outputs[0].data_;
	Scale bottom_size = outputs[0].size_;
	int bottom_column_stride = bottom_size[0];
	int bottom_channel_stride = bottom_size[1]*bottom_size[0];
	int bottom_image_stride = bottom_size[2]*bottom_size[1]*bottom_size[0];

	element_t* filters = inputs[1].data_;
	Scale filtersize = inputs[1].size_;
	int filter_column_stride = filtersize[0];
	int filter_channel_stride = filtersize[1]*filtersize[0];
	int filter_element_stride = filtersize[2]*filtersize[1]*filtersize[0];

	//element_t* bias = inputs[2].data_;

	element_t* top_diff = inputs[0].data_;
	Scale top_size = inputs[0].size_;
	//int top_column_stride = top_size[0];
	//int top_channel_stride = top_size[1]*top_size[0];
	//int top_image_stride = top_size[2]*top_size[1]*top_size[0];

	for(int i = 0; i < bottom_size.Prod(); i++){
		bottom_diff[i] = 0.0f;
	}


    int pad_height = closure.pad_height;
    int pad_width = closure.pad_width;
    int stride_vertical = closure.stride_vertical;
    int stride_horizontal = closure.stride_horizontal;

    /*
    printf("bottom: (%d,%d) padding: (%d,%d), filter: (%d,%d), top: (%d,%d)\n",
    		bottom_size[0], bottom_size[1],
			pad_width, pad_height,
			filtersize[0], filtersize[1],
			top_size[0], top_size[1]);
*/
    int top_index = -1  ;

	for(int element = 0 ; element < bottom_size[3]; element++){
		for(int filter = 0; filter < filtersize[3]; filter++){
			int filter_offset = filter_element_stride*filter;
			for(int y = -pad_height; y <= bottom_size[1]-filtersize[1]+pad_height; y = y + stride_vertical){
				for(int x = -pad_width; x <= bottom_size[0]-filtersize[0]+pad_width; x += stride_horizontal){
					top_index++;
					for(int channel =0; channel < bottom_size[2]; channel++){
						for(int filter_y = 0; filter_y < filtersize[1]; filter_y++){
							for(int filter_x = 0; filter_x < filtersize[0]; filter_x++){
								if(x+filter_x >= 0  && y+filter_y >= 0 && x+filter_x < bottom_size[0] && y+filter_y < bottom_size[1]){
									int bottom_index = (x+filter_x)+bottom_column_stride*(y+filter_y)+bottom_channel_stride*channel+bottom_image_stride*element ;
									int filter_index = (filtersize[0] -1 -filter_x)+filter_column_stride*(filtersize[1] -1 -filter_y)+filter_channel_stride*channel+filter_offset;
									bottom_diff[bottom_index] += top_diff[top_index ] * filters[filter_index] ;
									/*
									if (bottom_index == 0){
										printf("top:(%d,%d) %d ; filter: d(%d,%d) (%d,%d) %d\n", \
												top_index%top_size[0], top_index/top_size[0], top_index, \
												filter_x, filter_y,(filtersize[0] -1 -filter_x), (filtersize[1] -1 -filter_y), filter_index);
									}
									*/
								}
							}
						}
					}//End full filter (filter_x*filter_y*channel)
				}
			}
		}
	}
}

/*
 * NOTE: This implementation is not designed to be fast.  If you want that, use a GPU.  It is meant to provide feature completeness and the ability to develop on a GPU-less machine.
 */
void ConvBackwardBias(const DataList& inputs, const DataList& outputs, ConvBackwardBiasClosure& closure){
	CHECK_EQ(inputs.size(), 1) << " Convolution #inputs are wrong.\n";
	CHECK_EQ(outputs.size(), 1) << " Convolution #outputs are wrong.\n";

	element_t* bottom_diff = outputs[0].data_;
	Scale bottom_size = outputs[0].size_;
	//int bottom_image_stride = bottom_size[0];
	//cout << "convolution bias_diff size " << bottom_size.ToString() << "\n";

	element_t* top_diff = inputs[0].data_;
	Scale top_size = inputs[0].size_;
	int top_column_stride = top_size[0];
	int top_channel_stride = top_size[1]*top_size[0];
	int top_image_stride = top_size[2]*top_size[1]*top_size[0];


	for(int i = 0; i < top_size[2]; i++){
		bottom_diff[i] = 0.0f;
	}

	for(int element = 0 ; element < top_size[3]; element++){
		//printf("element\n");
		for(int filter = 0; filter < top_size[2]; filter++){
			int bottom_index = filter; //+ element*bottom_image_stride;
			for(int y = 0; y < top_size[1]; y++){
				//printf("\trow %d / %d X %d\n",y,inputs[0].size_.get(1)-inputs[1].size_.get(1)+pad_height+1,stride_vertical);
				for(int x = 0; x < top_size[0]; x ++){
					//printf("\t\tcolumn %d / %d X %d\n",x,inputs[0].size_.get(0)-inputs[1].size_.get(0)+pad_width+1,stride_horizontal);
						int top_index = x + top_column_stride*y + top_channel_stride*filter + top_image_stride*element;
						bottom_diff[bottom_index] += top_diff[top_index];
				}
			}
		}
	}
}

/*
 * NOTE: This implementation is not designed to be fast.  If you want that, use a GPU.  It is meant to provide feature completeness and the ability to develop on a GPU-less machine.
 */
void ConvBackwardFilter(const DataList& inputs, const DataList& outputs, ConvBackwardFilterClosure& closure){
	CHECK_EQ(inputs.size(), 2) << " Convolution #inputs are wrong.\n";
	CHECK_EQ(outputs.size(), 1) << " Convolution #outputs are wrong.\n";

	element_t* bottom = inputs[1].data_;
	Scale bottom_size = inputs[1].size_;
	int bottom_column_stride = bottom_size[0];
	int bottom_channel_stride = bottom_size[1]*bottom_size[0];
	int bottom_image_stride = bottom_size[2]*bottom_size[1]*bottom_size[0];

	element_t* filter_diff = outputs[0].data_;
	Scale filtersize = outputs[0].size_;
	int filter_column_stride = filtersize[0];
	int filter_channel_stride = filtersize[1]*filtersize[0];
	int filter_element_stride = filtersize[2]*filtersize[1]*filtersize[0];

	//element_t* bias = inputs[2].data_;

	element_t* top_diff = inputs[0].data_;
	Scale top_size = inputs[0].size_;
	//int top_column_stride = top_size[0];
	//int top_channel_stride = top_size[1]*top_size[0];
	//int top_image_stride = top_size[2]*top_size[1]*top_size[0];


	for(int i = 0; i < filtersize.Prod(); i++){
		filter_diff[i] = 0.0f;
	}


    int pad_height = closure.pad_height;
    int pad_width = closure.pad_width;
    int stride_vertical = closure.stride_vertical;
    int stride_horizontal = closure.stride_horizontal;

    int topindex = -1  ;

	for(int element = 0 ; element < bottom_size[3]; element++){
		for(int filter = 0; filter < filtersize[3]; filter++){
			int filter_offset = filter_element_stride*filter;
			for(int y = -pad_height; y <= bottom_size[1]-filtersize[1]+pad_height; y = y + stride_vertical){
				for(int x = -pad_width; x <= bottom_size[0]-filtersize[0]+pad_width; x += stride_horizontal){
					topindex++;
					for(int channel =0; channel < bottom_size[2]; channel++){
						for(int filter_y = 0; filter_y < filtersize[1]; filter_y++){
							for(int filter_x = 0; filter_x < filtersize[0]; filter_x++){
								if(x+filter_x >= 0 && y+filter_y >= 0 && x+filter_x < bottom_size[0] && y+filter_y < bottom_size[1]){
										int bottom_index = (x+filter_x)+bottom_column_stride*(y+filter_y)+bottom_channel_stride*channel+bottom_image_stride*element ;
										int filter_index = (filtersize[0] -1 -filter_x)+filter_column_stride*(filtersize[1] -1 -filter_y)+filter_channel_stride*channel+filter_offset;
										filter_diff[filter_index] += bottom[bottom_index]* top_diff[topindex ];
								}
							}
						}
					}//End full filter (filter_x*filter_y*channel)
				}
			}
		}
	}
}

/*
 * NOTE: This implementation is not designed to be fast.  If you want that, use a GPU.  It is meant to provide feature completeness and the ability to develop on a GPU-less machine.
 */
void PoolingForward(const DataList& inputs, const DataList& outputs, PoolingForwardClosure& closure){

	element_t* input = inputs[0].data_;
	Scale insize = inputs[0].size_;
	int in_column = insize[0];
	int in_channel = insize[1]*insize[0];
	int in_element = insize[2]*insize[1]*insize[0];

	element_t* activations = outputs[0].data_;
	Scale outsize = outputs[0].size_;
	int out_column = outsize[0];
	int out_channel = outsize[1]*outsize[0];
	int out_element = outsize[2]*outsize[1]*outsize[0];


    int pad_height = closure.pad_height;
    int pad_width = closure.pad_width;
    int stride_vertical = closure.stride_vertical;
    int stride_horizontal = closure.stride_horizontal;
    int height = closure.height;
    int width = closure.width;
    /*printf("pad_height: %d, pad_width: %d, stride_v: %d, stride_h: %d, height: %d, width: %d\n", \
    		pad_height, pad_width, stride_vertical, stride_horizontal, height, width);
     */

	for(int image = 0 ; image < insize[3]; image++){
		//printf("element\n");
		for(int y = -pad_height; y < insize[1]-height+pad_height+1; y += stride_vertical){
			//printf("\trow %d / %d X %d\n",y,inputs[0].size_.get(1)-inputs[1].size_.get(1)+pad_height+1,stride_vertical);
			for(int x = -pad_width; x < insize[0]-width+pad_width+1; x += stride_horizontal){
				//printf("\t\tcolumn %d / %d X %d\n",x,inputs[0].size_.get(0)-inputs[1].size_.get(0)+pad_width+1,stride_horizontal);
				for(int channel =0; channel < inputs[0].size_.get(2); channel++){
					int outindex = (x+pad_width)/stride_horizontal + out_column*((y+pad_height)/stride_vertical) + out_channel*channel + out_element*image;
					int sample_index = (x)+in_column*(y)+in_channel*channel+in_element*image;
					if(x > 0 && x < insize[0] && y > 0 && y < insize[1]){
						activations[outindex] = input[x+in_column*y+in_channel*channel+in_element*image];
					}else{
						activations[outindex] = 0.0f;
					}
					//printf("\t\t\t\tchannel %d / %d\n",channel,inputs[1].size_.get(3));
					for(int filter_y = 0; filter_y < height; filter_y++){
						for(int filter_x = 0; filter_x < width; filter_x++){
							sample_index = (x+filter_x)+in_column*(y+filter_y)+in_channel*channel+in_element*image;
							//printf("bounds test\n");
							if(x+filter_x >= 0 && x+filter_x < insize[0] && y+filter_y >= 0 && y+filter_y < insize[1]){
								////printf("\t\t\t\t\t\top x: %d, y: %d\n",filter_x,filter_y);
								if(activations[outindex ] < input[sample_index]){
									/*
									if(sample_index < 0 ||sample_index >= insize.Prod()){
										printf("===Pooling sample index error (%d) === \n",sample_index);
									}
									*/
									activations[outindex ] =input[sample_index];
								}
								/*
								printf("\t\t\t\t\t(c: %d , x: %d + %d , y: %d + %d) input:%f  => [ %d ] %f\n",channel,x,filter_x,y,filter_y,
										(float)input[(x+filter_x)+in_column*(y+filter_y)+in_channel*channel+in_element*image],
										x/stride_horizontal + out_column*(y/stride_vertical) + out_channel*channel + out_element*image ,
										(float)activations[x/stride_horizontal + out_column*(y/stride_vertical) + out_channel*channel + out_element*image ]);
								 */
							}
						}
					}
				}//End full filter (filter_x*filter_y*channel)
			}
		}
	}
}


/*
 * NOTE: This implementation is not designed to be fast.  If you want that, use a GPU.  It is meant to provide feature completeness and the ability to develop on a GPU-less machine.
 */
void PoolingBackward(const DataList& inputs, const DataList& outputs, PoolingBackwardClosure& closure){
	//TODO(Jesse Lovitt): Test this implementation.
	auto& top_diff = inputs[0];
	auto& top = inputs[1];
	auto& bottom = inputs[2];
	auto& bottom_diff = outputs[0];
	//int num_images = top_diff.size_[3];
	//int num_channels = top_diff.size_[2];
	//int bottom_height = bottom.size_[1];
	//int bottom_width = bottom.size_[0];

	Scale bottom_size = inputs[2].size_;
	int bottom_column_stride = bottom_size[0];
	int bottom_channel_stride = bottom_size[1]*bottom_size[0];
	int bottom_image_stride = bottom_size[2]*bottom_size[1]*bottom_size[0];

	Scale top_size = inputs[0].size_;
	int top_column_stride = top_size[0];
	int top_channel_stride = top_size[1]*top_size[0];
	int top_image_stride = top_size[2]*top_size[1]*top_size[0];


	int pad_height = closure.pad_height;
	int pad_width = closure.pad_width;
	int stride_vertical = closure.stride_vertical;
	int stride_horizontal = closure.stride_horizontal;
	int height = closure.height;
	int width = closure.width;

	for(int i = 0; i < bottom_size.Prod(); i++){
		bottom_diff.data_[i] = 0.0f;
	}


	for(int image = 0 ; image < bottom_size[3]; image++){
		//printf("element\n");
		for(int y = -pad_height; y < bottom_size[1]-height+pad_height+1; y = y + stride_vertical){
			//printf("\trow %d / %d X %d\n",y,inputs[0].size_.get(1)-inputs[1].size_.get(1)+pad_height+1,stride_vertical);
			for(int x = -pad_width; x < bottom_size[0]-width+pad_width+1; x += stride_horizontal){
				//printf("\t\tcolumn %d / %d X %d\n",x,inputs[0].size_.get(0)-inputs[1].size_.get(0)+pad_width+1,stride_horizontal);
				for(int channel =0; channel < inputs[0].size_.get(2); channel++){
					int top_index = (x+pad_width)/stride_horizontal + top_column_stride*((y+pad_height)/stride_vertical) + top_channel_stride*channel + top_image_stride*image;
					//printf("\t\t\t\tchannel %d / %d\n",channel,inputs[1].size_.get(3));
					for(int filter_y = 0; filter_y < height; filter_y++){
						for(int filter_x = 0; filter_x < width; filter_x++){
							if(x >= 0 && x+filter_x < bottom_size[0] && y >= 0 && y+filter_y < bottom_size[1]){
								////printf("\t\t\t\t\t\top x: %d, y: %d\n",filter_x,filter_y);
								int bottom_index = (x+filter_x)+bottom_column_stride*(y+filter_y)+bottom_channel_stride*channel+bottom_image_stride*image;
								//TODO(Jesse Lovitt): Caution... float comparison.
								//printf("top: %f[%d] vs bottom: %f[%d]\n",top.data_[top_index ],top_index, bottom.data_[bottom_index],bottom_index);
								if(top.data_[top_index ] == bottom.data_[bottom_index]){
									/*
									if((x+filter_x)+bottom_column_stride*(y+filter_y)+bottom_channel_stride*channel+bottom_image_stride*image < 0 ||(x+filter_x)+bottom_column_stride*(y+filter_y)+bottom_channel_stride*channel+bottom_image_stride*image >= bottom_size.Prod()){
										printf("Pooling index error");
									}
									*/
									/*
									printf("bottom_diff: (%d,%d) += %f (%d,%d) => %f\n",x+filter_x,y+filter_y,
											top_diff.data_[top_index],x+pad_width,y+pad_height,
											bottom_diff.data_[bottom_index] + top_diff.data_[top_index]);
									*/
									bottom_diff.data_[bottom_index] += top_diff.data_[top_index];
								}


							}
						}
					}
				}//End full filter (filter_x*filter_y*channel)
			}
		}
	}
}

void FillLRNScale(element_t* bottom_data, element_t* scale_data, Scale size, int local_size, element_t alpha){
	  int num_img = size[3];
	  int channel = size[2];
	  int width = size[1];
	  int height = size[0];

	  int channel_stride = width*height;
	  int img_stride = channel_stride*channel;

	  for(int i = 0; i < size.Prod(); i++){
	  	  scale_data[i] = 0;
	  }
	  for(int img = 0; img < num_img; img++){
		  for(int c = 0; c < channel; c++){
			  for(int h = 0; h < height; h++){
				  for(int w = 0; w < width; w++){
					  //Not sure about the bounds here. split in half(yes according to Krizhevsky paper)? What about odd/even?
					  for(int j = c -local_size/2; j <= c+local_size/2; j++){
						  if(j >=0 && j < channel){
							  scale_data[h + height*w + channel_stride*c + img_stride*img] +=
									  bottom_data[h + height*w + channel_stride*j + img_stride*img]*bottom_data[h + height*w + channel_stride*j + img_stride*img];
						  }
					  }
				  }
			  }
		  }
	  }
	  for(int i = 0; i < size.Prod(); i++){
	  	  scale_data[i] = (1.0+(alpha/local_size)*scale_data[i]);
	  }
}

void LRNForward(const DataList& inputs, const DataList& outputs, LRNForwardClosure& closure) {
  CHECK_EQ(inputs.size(), 2) << "(LRNForward) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(LRNForward) #outputs is wrong!";
  element_t* bottom_data = inputs[0].data_;
  element_t* scale_data = inputs[1].data_;
  element_t* res_data = outputs[0].data_;
  int local_size = closure.local_size;
  element_t alpha = closure.alpha;
  element_t beta = closure.beta;


  FillLRNScale(bottom_data, scale_data, inputs[0].size_, local_size, alpha);

  //res_data[i] = bottom_data[i] / pow((alpha * sum_squared_neighbors),beta)
  for(int i = 0; i < inputs[1].size_.Prod(); i++){
	  scale_data[i] = pow(scale_data[i],beta);
	  res_data[i] = bottom_data[i]/scale_data[i];
  }
}


void LRNBackward(const DataList& inputs, const DataList& outputs, LRNBackwardClosure& closure) {
  CHECK_EQ(inputs.size(), 4) << "(LRNBackward) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(LRNBackward) #outputs is wrong!";
  element_t* bottom_data = inputs[0].data_;
  element_t* top_data = inputs[1].data_;
  element_t* scale_data = inputs[2].data_;
  element_t* top_diff = inputs[3].data_;
  element_t* bottom_diff = outputs[0].data_;
  int local_size = closure.local_size;
  element_t alpha = closure.alpha;
  element_t beta = closure.beta;
/*  int num_img = closure.data_shape[3];
  int channel = closure.data_shape[2];
  int weight = closure.data_shape[1];
  int height = closure.data_shape[0];*/

  FillLRNScale(bottom_data, scale_data, inputs[0].size_, local_size, alpha);

  element_t cache_ratio = beta*2*alpha/local_size;

  for(int i = 0; i < inputs[0].size_.Prod(); i++){
	  bottom_diff[i] = top_diff[i]*( pow(scale_data[i],-beta) - cache_ratio*top_data[i]*bottom_data[i]/scale_data[i] );
  }
}


void Index(const DataList& inputs, const DataList& outputs, IndexClosure& closure) {
	CHECK_EQ(inputs.size(), 1) << "(activation forward) #inputs wrong";
	CHECK_EQ(outputs.size(), 1) << "(activation forward) #outputs wrong";
	element_t* input_data = inputs[0].data_;
	element_t* output_data = outputs[0].data_;

	size_t output_length = outputs[0].size_.Prod();
	auto idx = closure.idx;

	memcpy(output_data, input_data + idx * output_length, output_length * sizeof(input_data[0]));

	for (size_t i = 0; i < output_length; ++ i)
		output_data[i] = input_data[i + idx * output_length];
}

}  // end of namespace basic
}  // end of namespace minerva
