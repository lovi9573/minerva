/*
 * device_mpi.h
 *
 *  Created on: Jun 29, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_DEVICE_DEVICE_MPI_H_
#define MINERVA_DEVICE_DEVICE_MPI_H_

#include "device/device.h"

namespace minerva {


#ifdef HAS_MPI
class MpiDevice : public ThreadedDevice {
 public:
  MpiDevice( int rank, uint64_t device_id, DeviceListener*, int gpu_id);
  DISALLOW_COPY_AND_ASSIGN(MpiDevice);
  ~MpiDevice();
  MemType GetMemType() const override;
  std::string Name() const override;
  void FreeDataIfExist(uint64_t data_id) override;

 private:
  static size_t constexpr kParallelism = 4;
  int _gpu_id;
  void DoCopyRemoteData(element_t*, element_t*, size_t, int) override;
  void Execute(Task*, int thrid);
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) override;
  void DoExecute(Task* task, int thrid) override;
};
#endif //HAS_MPI

} //namespace minerva

#endif /* MINERVA_DEVICE_DEVICE_MPI_H_ */
