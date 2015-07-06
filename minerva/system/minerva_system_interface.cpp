

#include "system/minerva_system.h"




namespace minerva {

//TODO(jlovitt): Consider making a MinervaSystem interface to keep mpi bits hidden from owl.

IMinervaSystem& IMinervaSystem::Interface(){
	return MinervaSystem::Instance();
}

void IMinervaSystem::Init(int* argc, char*** argv){
	MinervaSystem::Initialize(argc, argv);
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

}  // end of namespace minerva

