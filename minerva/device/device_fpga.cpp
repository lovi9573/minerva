#include <utility>
#include <cstdlib>
#include <array>
#include <mutex>
#include <sstream>
#include <cstring>

#include <dmlc/logging.h>
#include <gflags/gflags.h>

#include "device.h"
#include "device/task.h"
#include "system/minerva_system.h"
#include "op/context.h"
#include "common/cuda_utils.h"
#include "device/pooled_data_store.h"
#include "profiler/wall_timer.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cuda.h>
#include <cublas_v2.h>
#endif


using namespace std;

namespace minerva {


#ifdef HAS_FPGA
/*
 * ====================
 * 		FPGA
 * ====================
 */
FpgaDevice::FpgaDevice(uint64_t device_id, DeviceListener*, int sub_id){
	  auto allocator = [](size_t len) -> void* {
	    void* ret = malloc(len);
	    return ret;
	  };
	  auto deallocator = [](void* ptr) {
	    free(ptr);
	  };
	  data_store_ = common::MakeUnique<DataStore>(allocator, deallocator);
}


FpgaDevice::~FpgaDevice(){
	  pool_.WaitForAllFinished();
}

MemType FpgaDevice::GetMemType() {
	return MemType::kFpga;
}

std::string FpgaDevice::Name(){
	return common::FString("FPGA device #%d", device_id_);
}

  //struct Impl;
  //std::unique_ptr<Impl> impl_;
  //void PreExecute() override;
  //void Barrier(int) override;
  void FpgaDevice::DoCopyRemoteData(float*, float*, size_t, int) {
	//TODO(jlovitt): How to copy in / out of fpga?
  }
  void FpgaDevice::DoExecute(const DataList&, const DataList&, PhysicalOp&, int){
	  Context ctx;
	  ctx.impl_type = ImplType::kFpga;
	  //TODO(jlovitt): Other needed FPGA context.
	  op.compute_fn->Execute(in, out, ctx);
  }
  void FpgaDevice::DoExecute(Task* task, int thrid) override;
};
#endif // HAS_FPGA

}  // namespace minerva

