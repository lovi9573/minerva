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

//TODO(jlovitt): Perhaps this will be possible with RDMA
void MinervaSystem::UniversalMemcpy(
    pair<Device::MemType, float*> to,
    pair<Device::MemType, float*> from,
    size_t size) {
#ifdef HAS_CUDA
  CUDA_CALL(cudaMemcpy(to.second, from.second, size, cudaMemcpyDefault));
#else
  CHECK_EQ(static_cast<int>(to.first), static_cast<int>(Device::MemType::kCpu));
  CHECK_EQ(static_cast<int>(from.first), static_cast<int>(Device::MemType::kCpu));
/*  printf("\nUniversal memcopy copying %lu floats at %p\n", size/sizeof(float), from.second);
	for(size_t i =0; i < size/sizeof(float); i ++){
		printf("%f, ", *(  ((float*)from.second)+i        ));
	}*/
  memcpy(to.second, from.second, size);
#endif
}




int const IMinervaSystem::has_cuda_ =
#ifdef HAS_CUDA
1
#else
0
#endif
;

int const IMinervaSystem::has_mpi_ =
#ifdef HAS_MPI
1
#else
0
#endif
;

int const IMinervaSystem::has_fpga_ =
#ifdef HAS_FPGA
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
	DLOG(INFO) << "cpu device creation in rank" << rank_ << "\n";
	if (worker_){
		LOG(FATAL) << "Cannot create a unique device id on worker rank " << rank_;
	}
  return MinervaSystem::Instance().device_manager().CreateCpuDevice();
}

uint64_t MinervaSystem::CreateGpuDevice(int id) {
	if (worker_){
		LOG(FATAL) << "Cannot create a unique device id on worker rank";
	}
  return MinervaSystem::Instance().device_manager().CreateGpuDevice(id);
}

uint64_t MinervaSystem::CreateFpgaDevice(int sub_id ) {
	if (worker_){
		LOG(FATAL) << "Cannot create a unique device id on worker rank";
	}
	return  MinervaSystem::Instance().device_manager().CreateFpgaDevice(sub_id);
}

uint64_t MinervaSystem::CreateMpiDevice(int rank, int id ) {
	if (worker_){
		LOG(FATAL) << "Cannot create a unique device id on worker rank";
	}
	CHECK(has_mpi_) << "Cannot create MPI device.  Recompile with MPI support.";
	uint64_t device_id = UINT64_C(0xffffffffffffffff) ;
#ifdef HAS_MPI
	uint64_t device_id =  MinervaSystem::Instance().device_manager().CreateMpiDevice(rank,id);
	mpiserver_->CreateMpiDevice(rank, id, device_id);
#endif
	return device_id;
}


void MinervaSystem::SetDevice(uint64_t id) {
  current_device_id_ = id;
}
void MinervaSystem::WaitForAll() {
  backend_->WaitForAll();
}

int MinervaSystem::rank(){
	return rank_;
}

void MinervaSystem::WorkerRun(){
#ifdef HAS_MPI
	mpihandler_->MainLoop();
#endif
}

MinervaSystem::MinervaSystem(int* argc, char*** argv)
  : data_id_counter_(0), task_id_counter_(0), current_device_id_(0), rank_(0), worker_(false) {
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

	//MPI_Init(argc, argv);
	rank_ = ::MPI::COMM_WORLD.Get_rank();
	fflush(stdout);
	if(rank_ != 0){
		worker_ = true;
		mpihandler_ = new MpiHandler(rank_);
	}
	else{
		worker_ = false;
		mpiserver_ = new MpiServer();
		std::thread t(&MpiServer::MainLoop, mpiserver_);
		t.detach();
	}
#endif
  physical_dag_ = new PhysicalDag();
  profiler_ = new ExecutionProfiler();
  device_manager_ = new DeviceManager(worker_);
  if (FLAGS_use_dag && !worker_) {
    DLOG(INFO) << "dag engine enabled";
    backend_ = new DagScheduler(physical_dag_, device_manager_);
  } else {
    DLOG(INFO) << "dag engine disabled";
    backend_ = new SimpleBackend(*device_manager_);
  }
}



#ifdef HAS_MPI


void MinervaSystem::Request_Data(char* buffer, size_t bytes, int rank, uint64_t device_id, uint64_t data_id){
	if(worker_){
		mpihandler_->Request_Data(buffer, bytes, rank, device_id, data_id);
	}else{
		mpiserver_->Request_Data(buffer, bytes, rank, device_id, data_id);
	}
}

#endif //end HAS_MPI


}  // end of namespace minerva

