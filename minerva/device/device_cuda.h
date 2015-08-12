/*
 * device_cuda.h
 *
 *  Created on: Jun 29, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_DEVICE_DEVICE_CUDA_H_
#define MINERVA_DEVICE_DEVICE_CUDA_H_

#include "device/device.h"

namespace minerva {

#ifdef HAS_CUDA
class GpuDevice : public ThreadedDevice {
 public:
  GpuDevice(uint64_t device_id, DeviceListener*, int gpu_id);
#ifdef HAS_MPI
  GpuDevice(int rank, uint64_t device_id, DeviceListener*, int gpu_id);
#endif
  DISALLOW_COPY_AND_ASSIGN(GpuDevice);
  ~GpuDevice();
  MemType GetMemType() const override;
  std::string Name() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  void PreExecute() override;
  void Barrier(int) override;
  void DoCopyRemoteData(element_t*, element_t*, size_t, int) override;
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) override;
//  void DoExecute(Task* task, int thrid) override;
};
#endif


} // namespace minerva

#endif /* MINERVA_DEVICE_DEVICE_CUDA_H_ */
