#pragma once
#include <iostream>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cublas.h>
#include <cudnn.h>
#endif
#ifdef HAS_FPGA
#include "Ht.h"
#endif

namespace minerva {

enum class ImplType {
  kNA = 0,
  kBasic,
  kMkl,
  kCuda,
  kFpga,
  kMpi
};

inline std::ostream& operator<<(std::ostream& os, ImplType t) {
  switch (t) {
    case ImplType::kNA: return os << "N/A";
    case ImplType::kBasic: return os << "Basic";
    case ImplType::kMkl: return os << "Mkl";
    case ImplType::kCuda: return os << "Cuda";
    case ImplType::kFpga: return os << "FPGA";
    default: return os << "Unknown impl type";
  }
}

struct Context {
  ImplType impl_type;
#ifdef HAS_CUDA
  cudaStream_t stream;
  cublasHandle_t cublas_handle;
  cudnnHandle_t cudnn_handle;
#endif
#ifdef HAS_MPI
  int rank;
  int gpu_id;
#endif
#ifdef HAS_FPGA
  CHtHif *pHt_host_interface;
  CHtAuUnit ** pAuUnits;
  int unitCnt;
#endif
  virtual ~Context() {
  };
};

}

