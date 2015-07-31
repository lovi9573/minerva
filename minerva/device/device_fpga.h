/*
 * device_fpga.h
 *
 *  Created on: Jun 29, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_DEVICE_DEVICE_FPGA_H_
#define MINERVA_DEVICE_DEVICE_FPGA_H_

#ifdef HAS_FPGA
#include "device/device.h"
//#include "Ht.h"

namespace minerva {

class FpgaDevice : public ThreadedDevice {
 public:
  FpgaDevice(uint64_t device_id, DeviceListener*, int sub_id);
  DISALLOW_COPY_AND_ASSIGN(FpgaDevice);
  ~FpgaDevice();
  Device::MemType GetMemType() const override;
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
  void DoCopyRemoteData(element_t*, element_t*, size_t, int) override;
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) override;
  void DoExecute(Task* task, int thrid) override;
//  CHtHif *pHt_host_interface;
//  CHtAuUnit ** pAuUnits;
//  int unitCnt_;
};

} //namespace minerva
#endif // HAS_FPGA

#endif /* MINERVA_DEVICE_DEVICE_FPGA_H_ */
