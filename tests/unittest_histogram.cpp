#include "unittest_main.h"
#include <cmath>

using namespace std;
using namespace minerva;

TEST(Elewise, CpuExp) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  float input_raw[] = {1,2,3,4,5,6,7,8,9};
  Scale size{3, 3};
  shared_ptr<float> input_ptr(new float[size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, size.Prod() * sizeof(float));

  int bins = 3;
  NArray input = NArray::MakeNArray(size,input_ptr);
  NArray out = input.Histogram(bins);
  auto output_ptr  = out.Get();
  for (int i = 0; i < bins; ++i) {
    //EXPECT_FLOAT_EQ(b_ptr.get()[i], exp(a_ptr.get()[i]));
	  printf("%f : %f",output_ptr.get()[i],output_ptr.get()[i+bins]);
  }
}

