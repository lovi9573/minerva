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
	// Get interfaces
	pHt_host_interface = new CHtHif();
	int unitCnt = pHt_host_interface->GetUnitCnt();
	printf("#AUs = %d\n", unitCnt);

	pAuUnits = new CHtAuUnit * [unitCnt];
	for (int unit = 0; unit < unitCnt; unit++){
		pAuUnits[unit] = new CHtAuUnit(pHt_host_interface);
	}

	/*
	auto allocator = pHt_host_interface->MemAlloc;
	auto deallocator = pHt_host_interface->MemFree;
	*/
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


void FpgaDevice::DoCopyRemoteData(float* dst, float* src, size_t size, int) {
	//pHt_host_interface->MemCpy(dst, src, size);
#ifdef HAS_CUDA
  CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
#else
  memcpy(dst, src, size);
#endif
}

void FpgaDevice::DoExecute(const DataList&, const DataList&, PhysicalOp&, int){
  Context ctx;
  ctx.impl_type = ImplType::kFpga;
  ctx.pHT_interface = pHt_host_interface;
  ctx.pAuUnits = pAuUnits;
  op.compute_fn->Execute(in, out, ctx);
}

void FpgaDevice::DoExecute(Task* task, int thrid) {
	LOG(FATAL) << "Cannot call DoExecute with Task* parameter on a CPU Device.";
}

#endif // HAS_FPGA

}  // namespace minerva

