#include "minerva_system.h"
#include <cstdlib>
#include <mutex>
#include <cstring>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif
#include <dmlc/logging.h>
#include <gflags/gflags.h>
#include "backend/dag/dag_scheduler.h"
#include "backend/simple_backend.h"
#include "common/cuda_utils.h"
#include <stdio.h>
#ifdef HAS_MPI
#include "mpi/mpi_handler.h"
#include "mpi/mpi_server.h"
#endif


DEFINE_bool(use_dag, false, "Use dag engine");
DEFINE_bool(no_init_glog, false, "Skip initializing Google Logging");

using namespace std;

namespace minerva {

//TODO: 1 multiway memcopy!
void MinervaSystem::UniversalMemcpy(
    pair<Device::MemType, float*> to,
    pair<Device::MemType, float*> from,
    size_t size) {
#ifdef HAS_CUDA
  CUDA_CALL(cudaMemcpy(to.second, from.second, size, cudaMemcpyDefault));
#else
  CHECK_EQ(static_cast<int>(to.first), static_cast<int>(Device::MemType::kCpu));
  CHECK_EQ(static_cast<int>(from.first), static_cast<int>(Device::MemType::kCpu));
  memcpy(to.second, from.second, size);
#endif
}




int const MinervaSystem::has_cuda_ =
#ifdef HAS_CUDA
1
#else
0
#endif
;

int const MinervaSystem::has_mpi_ =
#ifdef HAS_MPI
1
#else
0
#endif
;

MinervaSystem::~MinervaSystem() {
  delete backend_;
  delete device_manager_;
  delete profiler_;
  delete physical_dag_;
  //google::ShutdownGoogleLogging(); //XXX comment out since we switch to dmlc/logging
}

//TODO: 1 Translate symbolic mpi ptr ids to local data ptr.
pair<Device::MemType, float*> MinervaSystem::GetPtr(uint64_t device_id, uint64_t data_id) {
  return device_manager_->GetDevice(device_id)->GetPtr(data_id);
}

uint64_t MinervaSystem::GenerateDataId() {
  return data_id_counter_++;
}

uint64_t MinervaSystem::GenerateTaskId() {
  return task_id_counter_++;
}


uint64_t MinervaSystem::CreateCpuDevice() {
	LOG(INFO) << "cpu device creation in rank" << _rank << "\n";
	if (_worker){
		LOG(FATAL) << "Cannot create a unique device id on worker rank " << _rank;
	}
  return MinervaSystem::Instance().device_manager().CreateCpuDevice();
}
uint64_t MinervaSystem::CreateGpuDevice(int id) {
	if (_worker){
		LOG(FATAL) << "Cannot create a unique device id on worker rank";
	}
  return MinervaSystem::Instance().device_manager().CreateGpuDevice(id);
}

//TODO: 3 map these non-worker device functions into owl.
#ifdef HAS_MPI

uint64_t MinervaSystem::CreateMpiDevice(int rank, int id ) {
	if (_worker){
		LOG(FATAL) << "Cannot create a unique device id on worker rank";
	}
	uint64_t device_id =  MinervaSystem::Instance().device_manager().CreateMpiDevice(rank,id);
	mpiserver_->CreateMpiDevice(rank, id, device_id);
	return device_id;
}

#endif //end HAS_MPI

void MinervaSystem::SetDevice(uint64_t id) {
  current_device_id_ = id;
}
void MinervaSystem::WaitForAll() {
  backend_->WaitForAll();
}

int MinervaSystem::rank(){
#ifdef HAS_MPI
	return _rank;
#else
	return 0;
#endif
}

void MinervaSystem::WorkerRun(){
	mpihandler_->MainLoop();
}

void MinervaSystem::Request_Data(char* buffer, size_t bytes, int rank, uint64_t device_id, uint64_t data_id){
	if(_worker){
		mpihandler_->Request_Data(buffer, bytes, rank, device_id, data_id);
	}else{
		mpiserver_->Request_Data(buffer, bytes, rank, device_id, data_id);
	}
}

MinervaSystem::MinervaSystem(int* argc, char*** argv)
  : data_id_counter_(0), task_id_counter_(0), current_device_id_(0), _worker(false) {
  gflags::ParseCommandLineFlags(argc, argv, true);
#ifndef HAS_PS
  // glog is initialized in PS::main, and also here, so we will hit a
  // double-initalize error when compiling with PS
  if (!FLAGS_no_init_glog) {
    //google::InitGoogleLogging((*argv)[0]); // XXX comment out since we switch to dmlc/logging
  }
#endif
#ifdef HAS_MPI
  	int provided;
  	MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
	if( provided != MPI_THREAD_MULTIPLE){
		LOG(FATAL) << "Multithreaded mpi support is needed.\n";
	}
	_rank = ::MPI::COMM_WORLD.Get_rank();
	fflush(stdout);
	if(_rank != 0){
		_worker = true;
		mpihandler_ = new MpiHandler(_rank);
	}
	else{
		_worker = false;
		mpiserver_ = new MpiServer();
	}
#endif
  physical_dag_ = new PhysicalDag();
  profiler_ = new ExecutionProfiler();
  device_manager_ = new DeviceManager(_worker);
  if (FLAGS_use_dag && !_worker) {
    LOG(INFO) << "dag engine enabled";
    backend_ = new DagScheduler(physical_dag_, device_manager_);
  } else {
    LOG(INFO) << "dag engine disabled";
    backend_ = new SimpleBackend(*device_manager_);
  }
}




}  // end of namespace minerva

