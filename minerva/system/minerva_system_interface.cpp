

#include "system/minerva_system.h"




namespace minerva {

//TODO(jlovitt): Consider making a MinervaSystem interface to keep mpi bits hidden from owl.

IMinervaSystem& IMinervaSystem::Interface(){
	return MinervaSystem::Instance();
}

void IMinervaSystem::Init(int* argc, char*** argv){
	MinervaSystem::Initialize(argc, argv);
}


}  // end of namespace minerva

