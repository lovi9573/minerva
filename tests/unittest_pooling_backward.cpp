#include "unittest_main.h"

using namespace std;
using namespace minerva;

#ifdef HAS_CUDA
TEST(PoolingForward, DISABLED_GpuWithoutPadding) {
  float input_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float correct_raw[] = {11, 12, 15, 16};
  auto& ms = MinervaSystem::Instance();
  Scale input_size{4, 4, 1, 1};
  Scale correct_size{2, 2, 1, 1};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  ms.SetDevice(gpu_device);
  ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 1, 1);
  ImageBatch output = Convolution::PoolingForward(input, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}

TEST(PoolingForward, DISABLED_GpuWithExactPadding) {
  float input_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float correct_raw[] = {6, 7, 8, 8, 10, 11, 12, 12, 14, 15, 16, 16, 14, 15, 16, 16};
  auto& ms = MinervaSystem::Instance();
  Scale input_size{4, 4, 1, 1};
  Scale correct_size{4, 4, 1, 1};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  ms.SetDevice(gpu_device);
  ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 1, 1, 1, 1);
  ImageBatch output = Convolution::PoolingForward(input, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}

TEST(PoolingForward, DISABLED_GpuWithInsufficientPadding) {
  float input_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float correct_raw[] = {6, 8, 8, 14, 16, 16, 14, 16, 16};
  auto& ms = MinervaSystem::Instance();
  Scale input_size{4, 4, 1, 1};
  Scale correct_size{3, 3, 1, 1};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  ms.SetDevice(gpu_device);
  ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 2, 2, 1, 1);
  ImageBatch output = Convolution::PoolingForward(input, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}

TEST(PoolingForward, DISABLED_GpuWithTooMuchPadding) {
  float input_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float correct_raw[] = {6, 8, 14, 16};
  auto& ms = MinervaSystem::Instance();
  Scale input_size{4, 4, 1, 1};
  Scale correct_size{2, 2, 1, 1};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  ms.SetDevice(gpu_device);
  ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 4, 4, 3, 3, 2, 2);
  ImageBatch output = Convolution::PoolingForward(input, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}

#endif

TEST(PoolingForward, CpuWithoutPadding) {
  float top_raw[] = {5,6,18,18};
  float top_diff_raw[] = {1.0,1.1,1.2,1.3};
  float bottom_raw[] = {1,2,3,4,5,6,7,18,9};
  float correct_raw[] = {0,0,0,0,1.0,1.1,0,2.5,0};
  Scale top_size{2, 2, 1, 1};
  Scale correct_size{3, 3, 1, 1};
  shared_ptr<float> top_ptr(new float[top_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_diff_ptr(new float[top_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> bottom_ptr(new float[correct_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(top_ptr.get(), top_raw, top_size.Prod() * sizeof(float));
  memcpy(top_diff_ptr.get(), top_diff_raw, top_size.Prod() * sizeof(float));
  memcpy(bottom_ptr.get(), bottom_raw, correct_size.Prod() * sizeof(float));
  ImageBatch top = NArray::MakeNArray(top_size, top_ptr);
  ImageBatch top_diff = NArray::MakeNArray(top_size, top_diff_ptr);
  ImageBatch bottom = NArray::MakeNArray(correct_size, bottom_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 2, 2, 1, 1);

  ImageBatch output = Convolution::PoolingBackward(top_diff, top, bottom, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}

TEST(PoolingForward, DISABLED_CpuWithExactPadding) {
  float input_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float correct_raw[] = {6, 7, 8, 8, 10, 11, 12, 12, 14, 15, 16, 16, 14, 15, 16, 16};
  Scale input_size{4, 4, 1, 1};
  Scale correct_size{4, 4, 1, 1};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 1, 1, 1, 1);
  ImageBatch output = Convolution::PoolingForward(input, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001) <<"index: " << i<< "\n";
  }
}

TEST(PoolingForward, DISABLED_CpuWithInsufficientPadding) {
  float input_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float correct_raw[] = {6, 8, 8, 14, 16, 16, 14, 16, 16};
  Scale input_size{4, 4, 1, 1};
  Scale correct_size{3, 3, 1, 1};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 2, 2, 1, 1);
  ImageBatch output = Convolution::PoolingForward(input, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}

TEST(PoolingForward, DISABLED_CpuWithTooMuchPadding) {
  float input_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float correct_raw[] = {6, 8, 14, 16};
  Scale input_size{4, 4, 1, 1};
  Scale correct_size{2, 2, 1, 1};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 4, 4, 3, 3, 2, 2);
  ImageBatch output = Convolution::PoolingForward(input, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}
