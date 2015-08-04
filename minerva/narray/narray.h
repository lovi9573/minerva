#pragma once
#include <initializer_list>
#include <dmlc/logging.h>
#include <memory>
#include "op/closure.h"
#include "common/scale.h"
#include "common/common.h"
#include "backend/backend.h"
#include "backend/backend_chunk.h"

namespace minerva {

struct FileFormat {
  bool binary;
};

class NArray {
  friend class Elewise;
  friend class Convolution;

 public:
  // Static constructors
  static NArray Constant(const Scale& size, element_t val);
  static NArray Randn(const Scale& size, float mu, float var);
  static NArray RandBernoulli(const Scale& size, float p);
  static NArray Zeros(const Scale& size);
  static NArray Ones(const Scale& size);
  static NArray MakeNArray(const Scale& size, std::shared_ptr<element_t> array);
  static NArray PushGradAndPullWeight(const NArray& grad, const std::string& layer_name);
  // DAG generating operations
  static std::vector<NArray> Compute(
      const std::vector<NArray>& params,
      const std::vector<Scale>& result_sizes,
      ComputeFn* fn);
  static NArray ComputeOne(
      const std::vector<NArray>& params,
      const Scale& size,
      ComputeFn* fn);
  static NArray GenerateOne(
      const Scale& size,
      ComputeFn* fn);
  // Constructors and destructors
  NArray();
  NArray(const NArray&);
  NArray(NArray&&);
  NArray& operator=(const NArray&);
  NArray& operator=(NArray&&);
  virtual ~NArray();
  // Element-wise operations
  friend NArray operator+(const NArray&, const NArray&);
  friend NArray operator-(const NArray&, const NArray&);
  friend NArray operator/(const NArray&, const NArray&);
  friend NArray operator+(element_t, const NArray&);
  friend NArray operator-(element_t, const NArray&);
  friend NArray operator*(element_t, const NArray&);
  friend NArray operator/(element_t, const NArray&);
  friend NArray operator+(const NArray&, element_t);
  friend NArray operator-(const NArray&, element_t);
  friend NArray operator*(const NArray&, element_t);
  friend NArray operator/(const NArray&, element_t);
  NArray& operator+=(const NArray&);
  NArray& operator-=(const NArray&);
  NArray& operator/=(const NArray&);
  NArray& operator+=(element_t);
  NArray& operator-=(element_t);
  NArray& operator*=(element_t);
  NArray& operator/=(element_t);
  NArray operator-() const;
  NArray operator[](int);
  // Matmult
  friend NArray operator*(const NArray&, const NArray&);
  NArray& operator*=(const NArray&);
  // Parameter server interaction
  NArray& Pull(const std::string& layer_name);
  // Concat
  friend NArray Concat(const std::vector<NArray>& params, int concat_dim);
  friend NArray Slice(const NArray& src, int slice_dim, int st_off, int slice_count);
  // Shape
  const Scale& Size() const { return CHECK_NOTNULL(data_)->shape(); }
  int Size(int dim) const { return CHECK_NOTNULL(data_)->shape()[dim]; }
  NArray Reshape(const Scale& dims) const;
  NArray Trans() const;
  NArray Select(std::vector<int> const&) const;
  // Lazy reductions
  NArray Sum(int dim) const;
  NArray Sum(const Scale& dims) const;
  NArray Max(int dim) const;
  NArray Max(const Scale& dims) const;
  NArray MaxIndex(int dim) const;

  // Replicate matrix
  NArray NormArithmetic(const NArray&, ArithmeticType) const;
  // Non-lazy reductions
  element_t Sum() const;  // TODO
  element_t Max() const;  // TODO
  int CountZero() const;
  // System
  void Wait() const;
  std::shared_ptr<element_t> Get() const;
  void ToStream(std::ostream& out, const FileFormat& format) const;
  void ToFile(const std::string& filename, const FileFormat& format) const;

 private:
  NArray(BackendChunk*);
  BackendChunk* data_;
};

// Matmult
NArray operator*(const NArray&, const NArray&);
NArray Concat(const std::vector<NArray>& params, int concat_dim);
NArray Slice(const NArray& src, int slice_dim, int st_off, int slice_count);

}  // end of namespace minerva

