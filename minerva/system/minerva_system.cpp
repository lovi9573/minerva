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
#include <dlfcn.h>
#include <mpi.h>
#include "mpi/mpi_handler.h"
#include "mpi/mpi_server.h"
#endif


DEFINE_bool(use_dag, true, "Use dag engine");
DEFINE_bool(no_init_glog, false, "Skip initializing Google Logging");

using namespace std;

namespace minerva {

//TODO(jlovitt): Perhaps this will be possible with RDMA
//TODO(jlovitt): Make this work with FPGA
void MinervaSystem::UniversalMemcpy(
    pair<Device::MemType, element_t*> to,
    pair<Device::MemType, element_t*> from,
    size_t size) {
#ifdef HAS_CUDA
  CUDA_CALL(cudaMemcpy(to.second, from.second, size, cudaMemcpyDefault));
#else
  CHECK_EQ(static_cast<int>(to.first), static_cast<int>(Device::MemType::kCpu));
  CHECK_EQ(static_cast<int>(from.first), static_cast<int>(Device::MemType::kCpu));
/*  printf("\nUniversal memcopy copying %lu element_ts at %p\n", size/sizeof(element_t), from.second);
	for(size_t i =0; i < size/sizeof(element_t); i ++){
		printf("%f, ", *(  ((element_t*)from.second)+i        ));
	}*/
  memcpy(to.second, from.second, size);
#endif
}






MinervaSystem::~MinervaSystem() {
  delete backend_;
  delete device_manager_;
  delete profiler_;
  delete physical_dag_;
#ifdef HAS_MPI
  if(!worker_){
	  mpiserver_->MPI_Terminate();
  }
#endif
  //google::ShutdownGoogleLogging(); //XXX comment out since we switch to dmlc/logging
}

pair<Device::MemType, element_t*> MinervaSystem::GetPtr(uint64_t device_id, uint64_t data_id) {
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
	device_id =  MinervaSystem::Instance().device_manager().CreateMpiDevice(rank,id);
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
  	//Workaround for openMpi's failure to load modules.  Documented at the bottom of this file.
  	dlopen("libmpi.so", RTLD_NOW | RTLD_GLOBAL);

  	int provided;
  	MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
	if( provided != MPI_THREAD_SERIALIZED){
		LOG(FATAL) << "Thread Serialized mpi support is needed.\n";
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
  if(!worker_){
	  if (FLAGS_use_dag) {
		DLOG(INFO) << "dag engine enabled";
		backend_ = new DagScheduler(physical_dag_, device_manager_);
	  } else {
		DLOG(INFO) << "dag engine disabled";
		backend_ = new SimpleBackend(*device_manager_);
	  }
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
/*
 * http://www.open-mpi.org/faq/?category=troubleshooting#missing-symbols

Open MPI loads a lot of plugins at run time. It opens its plugins via the excellent GNU Libtool libltdl portability library. Sometimes a plugin can fail to load because it can't resolve all the symbols that it needs. There are a few reasons why this can happen.

    The plugin is for a different version of Open MPI. See this FAQ entry for an explanation of how Open MPI might try to open the "wrong" plugins.
    An application is trying to manually dynamically open libmpi in a private symbol space. For example, if an application is not linked against libmpi, but rather calls something like this:

     This is a Linux example -- the issue is similar/the same on other
       operating systems
    handle = dlopen("libmpi.so", RTLD_NOW | RTLD_LOCAL);

    This is due to some deep run time linker voodoo -- it is discussed towards the end of this post to the Open MPI developer's list. Briefly, the issue is this:

        The dynamic library libmpi is opened in a "local" symbol space.
        MPI_INIT is invoked, which tries to open Open MPI's plugins.
        Open MPI's plugins rely on symbols in libmpi (and other Open MPI support libraries); these symbols must be resolved when the plugin is loaded.
        However, since libmpi was opened in a "local" symbol space, its symbols are not available to the plugins that it opens.
        Hence, the plugin fails to load because it can't resolve all of its symbols, and displays a warning message to that effect.

    The ultimate fix for this issue is a bit bigger than Open MPI, unfortunately -- it's a POSIX issue (as briefly described in the devel posting, above).

    However, there are several common workarounds:

        Dynamically open libmpi in a public / global symbol scope -- not a private / local scope. This will enable libmpi's symbols to be available for resolution when Open MPI dynamically opens its plugins.
        If libmpi is opened as part of some underlying framework where it is not possible to change the private / local scope to a public / global scope, then dynamically open libmpi in a public / global scope before invoking the underlying framework. This sounds a little gross (and it is), but at least the run-time linker is smart enough to not load libmpi twice -- but it does keeps libmpi in a public scope.
        Use the --disable-dlopen or --disable-mca-dso options to Open MPI's configure script (see this FAQ entry for more details on these options). These options slurp all of Open MPI's plugins up in to libmpi -- meaning that the plugins physically reside in libmpi and will not be dynamically opened at run time.
        Build Open MPI as a static library by configuring Open MPI with --disable-shared and --enable-static. This has the same effect as --disable-dlopen, but it also makes libmpi.a (as opposed to a shared library).
*/

/*
 *http://www.open-mpi.org/faq/?category=building#avoid-dso

 7. Can I disable Open MPI's use of plugins?

Yes.

Open MPI uses plugins for much of its functionality. Specifically, Open MPI looks for and loads plugins as dynamically shared objects (DSOs) during the call to MPI_INIT. However, these plugins can be compiled and installed in several different ways:

    As DSOs: In this mode (the default), each of Open MPI's plugins are compiled as a separate DSO that is dynamically loaded at run time.
        Advantage: this approach is highly flexible -- it gives system developers and administrators fine-grained approach to install new plugins to an existing Open MPI installation, and also allows the removal of old plugins (i.e., forcibly disallowing the use of specific plugins) simply by removing the corresponding DSO(s).
        Disadvantage: this approach causes additional filesystem traffic (mostly during MPI_INIT). If Open MPI is installed on a networked filesystem, this can cause noticable network traffic when a large parallel job starts, for example.

    As part of a larger library: In this mode, Open MPI "slurps up" the plugins includes them in libmpi (and other libraries). Hence, all plugins are included in the main Open MPI libraries that are loaded by the system linker before an MPI process even starts.
        Advantage: Significantly less filesystem traffic than the DSO approach. This model can be much more performant on network installations of Open MPI.
        Disadvantage: Much less flexible than the DSO approach; system administrators and developers have significantly less ability to add/remove plugins from the Open MPI installation at run-time. Note that you still have some ability to add/remove plugins (see below), but there are limitations to what can be done.

To be clear: Open MPI's plugins can be built either as standalone DSOs or included in Open MPI's main libraries (e.g., libmpi). Additionally, Open MPI's main libraries can be built either as static or shared libraries.

You can therefore choose to build Open MPI in one of several different ways:

    --disable-mca-dso: Using the --disable-mca-dso switch to Open MPI's configure script will cause all plugins to be built as part of Open MPI's main libraries -- they will not be built as standalone DSOs. However, Open MPI will still look for DSOs in the filesystem at run-time. Specifically: this option significantly decreases (but does not eliminate) filesystem traffic during MPI_INIT, but does allow the flexibility of adding new plugins to an existing Open MPI installation.

    Note that the --disable-mca-dso option does not affect whether Open MPI's main libraries are built as static or shared.

    --enable-static: Using this option to Open MPI's configure script will cause the building of static libraries (e.g., libmpi.a). This option automatically implies --disable-mca-dso.

    Note that --enable-shared is also the default; so if you use --enable-static, Open MPI will build both static and shared libraries that contain all of Open MPI's plugins (i.e., libmpi.so and libmpi.a). If you want only static libraries (that contain all of Open MPI's plugins), be sure to also use --disable-shared.

    --disable-dlopen: Using this option to Open MPI's configure script will do two things:
        Imply --disable-mca-dso, meaning that all plugins will be slurped into Open MPI's libraries.
        Cause Open MPI to not look for / open any DSOs at run time.

    Specifically: this option makes Open MPI not incur any additional filesystem traffic during MPI_INIT. Note that the --disable-dlopen option does not affect whether Open MPI's main libraries are built as static or shared.
 */
