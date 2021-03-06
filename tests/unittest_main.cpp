#include "unittest_main.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>

uint64_t cpu_device;
#ifdef HAS_CUDA
uint64_t gpu_device;
#endif

using namespace minerva;

class MinervaTestEnvironment : public testing::Environment {
 public:
  MinervaTestEnvironment(int* argc, char*** argv) : argc(argc), argv(argv) {
  }
  void SetUp() {
    MinervaSystem::Initialize(argc, argv);
    time_t seed = time(NULL);
    std::cout <<"Random Seed: "<< seed << "\n";
    srand(seed);
    cpu_device = MinervaSystem::Instance().device_manager().CreateCpuDevice();
#ifdef HAS_CUDA
    gpu_device = MinervaSystem::Instance().device_manager().CreateGpuDevice(0);
#endif
  }
  void TearDown() {
  }

 private:
  int* argc;
  char*** argv;
};

int main(int argc, char** argv) {
  testing::AddGlobalTestEnvironment(new MinervaTestEnvironment(&argc, &argv));
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

