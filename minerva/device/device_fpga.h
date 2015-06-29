/*
 * device_fpga.h
 *
 *  Created on: Jun 29, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_DEVICE_DEVICE_FPGA_H_
#define MINERVA_DEVICE_DEVICE_FPGA_H_

#include "device/device.h"

namespace minerva {

#ifdef HAS_FPGA
class FpgaDevice : public ThreadedDevice {
 public:
  FpgaDevice(uint64_t device_id, DeviceListener*, int sub_id);
  DISALLOW_COPY_AND_ASSIGN(FpgaDevice);
  ~FpgaDevice();
  MemType GetMemType() const override;
  std::string Name() const override;
#ifdef HAS_MPI
	FpgaDevice(int rank, uint64_t device_id, DeviceListener*, int sub_id);
#endif

 private:
  //struct Impl;
  //std::unique_ptr<Impl> impl_;
  //void PreExecute() override;
  //void Barrier(int) override;
  static size_t constexpr kParallelism = 4;
  void DoCopyRemoteData(float*, float*, size_t, int) override;
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) override;
  void DoExecute(Task* task, int thrid) override;
};
#endif // HAS_FPGA

} //namespace minerva

#endif /* MINERVA_DEVICE_DEVICE_FPGA_H_ */
