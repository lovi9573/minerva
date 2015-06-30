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
  float* left_data = inputs[0].data_;
  float* right_data = inputs[1].data_;
  float* res_data = outputs[0].data_;
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
  float val = closure.val;
  float* in_data = inputs[0].data_;
  float* res_data = outputs[0].data_;
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

void Elewise(const DataList& inputs, const DataList& outputs, ElewiseClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "(elewise) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(elewise) #outputs is wrong!";
  float* in_data = inputs[0].data_;
  float* res_data = outputs[0].data_;
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
  float* left_data = inputs[0].data_;
  float* right_data = inputs[1].data_;
  float* res_data = outputs[0].data_;
  int m = outputs[0].size_[0];
  int n = outputs[0].size_[1];
  int o = inputs[0].size_[1];
  // ATTENTION: the data is column major !!
#ifdef HAS_CBLAS
  memset(res_data, 0, sizeof(float) * m * n);
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
  float* in_data = inputs[0].data_;
  float* res_data = outputs[0].data_;
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
  float* in_data = inputs[0].data_;
  float* res_data = outputs[0].data_;
  auto in_max = inputs[0].size_;
  auto in_range = ScaleRange::MakeRangeFromOrigin(in_max);
  auto res_max = outputs[0].size_;
  auto res_range = ScaleRange::MakeRangeFromOrigin(res_max);
  auto accumulator = Scale::Origin(in_max.NumDims());
  do {
    auto cur = accumulator;
    float tmp = in_data[in_range.Flatten(cur)];
    while (cur.IncrDimensions(in_max, closure.dims_to_reduce)) {
      float tmp2 = in_data[in_range.Flatten(cur)];
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
  float* in_data = inputs[0].data_;
  float* res_data = outputs[0].data_;
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
    float tmp = in_data[in_range.Flatten(cur)];
    //get max
    while (cur.IncrDimensions(in_max, dim_to_norm)) {
      float tmp2 = in_data[in_range.Flatten(cur)];
      if (tmp < tmp2) {
        tmp = tmp2;
      }
    }
    //exp(x - max), also sum the result
    cur = accumulator;
    float sum_exp = 0;
    do {
      res_data[in_range.Flatten(cur)] = expf(in_data[in_range.Flatten(cur)] - tmp);
      sum_exp += res_data[in_range.Flatten(cur)];
    }while (cur.IncrDimensions(in_max, dim_to_norm));

    //devide the sum
    cur = accumulator;
    do {
      res_data[in_range.Flatten(cur)] /= sum_exp; 
    }while (cur.IncrDimensions(in_max, dim_to_norm));
  } while (accumulator.IncrWithDimensionsFixed(res_max, dim_to_norm));
}





void ArrayLoader(const DataList& outputs, ArrayLoaderClosure& closure) {
  CHECK_EQ(outputs.size(), 1) << "(array loader) #outputs wrong";
  CHECK(closure.data) << "probably already executed";
  memcpy(outputs[0].data_, closure.data.get(), outputs[0].size_.Prod() * sizeof(float));
  closure.data.reset();
}

void Randn(const DataList& output, RandnClosure& closure) {
  CHECK_EQ(output.size(), 1) << "wrong number of randn output";
  int length = output[0].size_.Prod();
  float* data = output[0].data_;
  default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
  normal_distribution<float> distribution(closure.mu, closure.var);
  for (int i = 0; i < length; ++i) {
    data[i] = distribution(generator);
  }
}

void RandBernoulli(const DataList& outputs, RandBernoulliClosure& closure) {
  CHECK_EQ(outputs.size(), 1) << "(bernoulli) #outputs wrong";
  int length = outputs[0].size_.Prod();
  float* data = outputs[0].data_;
  default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
  bernoulli_distribution distribution(closure.p);
  for (int i = 0; i < length; ++i) {
    data[i] = distribution(generator);
  }
}

void Fill(const DataList& output, FillClosure& closure) {
  CHECK_EQ(output.size(), 1) << "wrong number of fill constant output";
  int length = output[0].size_.Prod();
  float* data = output[0].data_;
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
  memcpy(res_data, normalizee_data, normalizee_size.Prod() * sizeof(float));
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
    float cur = normalizer_data[normalizer_range.Flatten(iterator_normalizer)];
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
  float* in_data = inputs[0].data_;
  float* res_data = outputs[0].data_;
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
    float cur_max = in_data[offset];
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
  memcpy(outputs[0].data_, inputs[0].data_, inputs[0].size_.Prod() * sizeof(float));
}

void SigmoidForward(const DataList& inputs, const DataList& outputs, SigmoidForwardClosure&) {
  CHECK_EQ(inputs.size(), 1) << "sigmoid forward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "sigmoid forward #outputs wrong";

  float* input_data = inputs[0].data_;
  float* output_data = outputs[0].data_;

  size_t numbers = inputs[0].size_.Prod();

  for (size_t i = 0; i < numbers; i++) {
    output_data[i] = 1.0 / (1.0 + expf(-input_data[i]));
  }
}

void ReluForward(const DataList& inputs, const DataList& outputs, ReluForwardClosure&) {
  CHECK_EQ(inputs.size(), 1) << "relu forward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "relu forward #outputs wrong";

  float* input_data = inputs[0].data_;
  float* output_data = outputs[0].data_;

  size_t numbers = inputs[0].size_.Prod();

  for (size_t i = 0; i < numbers; i++) {
    output_data[i] = input_data[i] > 0 ? input_data[i] : 0;
  }
}

void TanhForward(const DataList& inputs, const DataList& outputs, TanhForwardClosure&) {
  CHECK_EQ(inputs.size(), 1) << "tanh forward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "tanh forward #outputs wrong";

  float* input_data = inputs[0].data_;
  float* output_data = outputs[0].data_;

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


void Index(const DataList& inputs, const DataList& outputs, IndexClosure& closure) {
	CHECK_EQ(inputs.size(), 1) << "(activation forward) #inputs wrong";
	CHECK_EQ(outputs.size(), 1) << "(activation forward) #outputs wrong";
	float* input_data = inputs[0].data_;
	float* output_data = outputs[0].data_;

	size_t output_length = outputs[0].size_.Prod();
	auto idx = closure.idx;

	memcpy(output_data, input_data + idx * output_length, output_length * sizeof(input_data[0]));
	for (size_t i = 0; i < output_length; ++ i)
		cout << output_data[i] << endl;

	for (size_t i = 0; i < output_length; ++ i)
		output_data[i] = input_data[i + idx * output_length];
}

}  // end of namespace basic
}  // end of namespace minerva

