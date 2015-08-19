#include "unittest_main.h"

using namespace std;
using namespace minerva;

#ifdef HAS_CUDA

TEST(PoolingBackward, GpuWithoutPadding) {
  float top_raw[] = {5,6,18,18};
  float top_diff_raw[] = {1.0,1.1,1.2,1.3};
  float bottom_raw[] = {1,2,3,4,5,6,7,18,9};
  float correct_raw[] = {0,0,0,0,1.0,1.1,0,2.5,0};
  auto& ms = MinervaSystem::Instance();
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
  ms.SetDevice(gpu_device);
  ImageBatch output = Convolution::PoolingBackward(top_diff, top, bottom, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}

TEST(PoolingBackward, GpuWithExactPadding) {
  float bottom_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float top_raw[] = {6, 7, 8, 8, 10, 11, 12, 12, 14, 15, 16, 16, 14, 15, 16, 16};
  float top_diff_raw[] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6};

  float expected_raw[] = {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.100000, 0.200000, 0.700000, 0.000000, 0.500000, 0.600000, 1.500000, 0.000000, 2.200000, 2.400000, 5.400000};

  auto& ms = MinervaSystem::Instance();
  Scale bottom_size{4, 4, 1, 1};
  Scale top_size{4, 4, 1, 1};
  Scale top_diff_size{4, 4, 1, 1};

  shared_ptr<float> bottom_ptr(new float[bottom_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_ptr(new float[top_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_diff_ptr(new float[top_diff_size.Prod()], [](float* ptr) { delete[] ptr; });

  memcpy(bottom_ptr.get(), bottom_raw, bottom_size.Prod() * sizeof(float));
  memcpy(top_ptr.get(), top_raw, top_size.Prod() * sizeof(float));
  memcpy(top_diff_ptr.get(), top_diff_raw, top_diff_size.Prod() * sizeof(float));

  ms.SetDevice(gpu_device);
  ImageBatch bottom = NArray::MakeNArray(bottom_size, bottom_ptr);
  ImageBatch top = NArray::MakeNArray(top_size, top_ptr);
  ImageBatch top_diff = NArray::MakeNArray(top_diff_size, top_diff_ptr);

  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 1, 1, 1, 1);
  ImageBatch output = Convolution::PoolingBackward(top_diff, top, bottom, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), bottom_size);
  for (int i = 0; i < bottom_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], expected_raw[i], 0.001);
	//printf("%f, ",output_ptr.get()[i]);
  }
}


TEST(PoolingBackward, GpuWithInsufficientPadding) {
  float bottom_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float top_raw[] = {6, 8, 8, 14, 16, 16, 14, 16, 16};
  float top_diff_raw[] = {0.5,0.4,0.1,0.9,1.2,0.3,1.4,1.5,2.1};
  float expected_raw[] = {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.500000, 0.000000, 0.500000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 2.300000, 0.000000, 5.100000};


  auto& ms = MinervaSystem::Instance();
  Scale bottom_size{4, 4, 1, 1};
  Scale top_size{3, 3, 1, 1};
  Scale top_diff_size{3, 3, 1, 1};

  shared_ptr<float> bottom_ptr(new float[bottom_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_ptr(new float[top_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_diff_ptr(new float[top_diff_size.Prod()], [](float* ptr) { delete[] ptr; });

  memcpy(bottom_ptr.get(), bottom_raw, bottom_size.Prod() * sizeof(float));
  memcpy(top_ptr.get(), top_raw, top_size.Prod() * sizeof(float));
  memcpy(top_diff_ptr.get(), top_diff_raw, top_diff_size.Prod() * sizeof(float));

  ms.SetDevice(gpu_device);
  ImageBatch bottom = NArray::MakeNArray(bottom_size, bottom_ptr);
  ImageBatch top = NArray::MakeNArray(top_size, top_ptr);
  ImageBatch top_diff = NArray::MakeNArray(top_diff_size, top_diff_ptr);

  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 2, 2, 1, 1);
  ImageBatch output = Convolution::PoolingBackward(top_diff, top, bottom, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), bottom_size);
  for (int i = 0; i < bottom_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], expected_raw[i], 0.001);
	//printf("%f, ",output_ptr.get()[i]);
  }
}

TEST(PoolingBackward, GpuWithTooMuchPadding) {
  float bottom_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float top_raw[] = {6, 8, 14, 16};
  float top_diff_raw[] = {0.3,1.4,1.5,2.1};
  float expected_raw[] = {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.300000, 0.000000, 1.400000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.500000, 0.000000, 2.100000};

  auto& ms = MinervaSystem::Instance();
  Scale bottom_size{4, 4, 1, 1};
  Scale top_size{2, 2, 1, 1};
  Scale top_diff_size{2, 2, 1, 1};

  shared_ptr<float> bottom_ptr(new float[bottom_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_ptr(new float[top_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_diff_ptr(new float[top_diff_size.Prod()], [](float* ptr) { delete[] ptr; });

  memcpy(bottom_ptr.get(), bottom_raw, bottom_size.Prod() * sizeof(float));
  memcpy(top_ptr.get(), top_raw, top_size.Prod() * sizeof(float));
  memcpy(top_diff_ptr.get(), top_diff_raw, top_diff_size.Prod() * sizeof(float));

  ms.SetDevice(gpu_device);
  ImageBatch bottom = NArray::MakeNArray(bottom_size, bottom_ptr);
  ImageBatch top = NArray::MakeNArray(top_size, top_ptr);
  ImageBatch top_diff = NArray::MakeNArray(top_diff_size, top_diff_ptr);

  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 4, 4, 3, 3, 2, 2);
  ImageBatch output = Convolution::PoolingBackward(top_diff, top, bottom, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), bottom_size);
  for (int i = 0; i < bottom_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], expected_raw[i], 0.001);
	//printf("%f, ",output_ptr.get()[i]);
  }
}

#endif

TEST(PoolingBackward, CpuWithoutPadding) {
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

TEST(PoolingBackward, CpuWithExactPadding) {
  float bottom_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float top_raw[] = {6, 7, 8, 8, 10, 11, 12, 12, 14, 15, 16, 16, 14, 15, 16, 16};
  float top_diff_raw[] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6};

  float expected_raw[] = {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.100000, 0.200000, 0.700000, 0.000000, 0.500000, 0.600000, 1.500000, 0.000000, 2.200000, 2.400000, 5.400000};


  Scale bottom_size{4, 4, 1, 1};
  Scale top_size{4, 4, 1, 1};
  Scale top_diff_size{4, 4, 1, 1};

  shared_ptr<float> bottom_ptr(new float[bottom_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_ptr(new float[top_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_diff_ptr(new float[top_diff_size.Prod()], [](float* ptr) { delete[] ptr; });

  memcpy(bottom_ptr.get(), bottom_raw, bottom_size.Prod() * sizeof(float));
  memcpy(top_ptr.get(), top_raw, top_size.Prod() * sizeof(float));
  memcpy(top_diff_ptr.get(), top_diff_raw, top_diff_size.Prod() * sizeof(float));


  ImageBatch bottom = NArray::MakeNArray(bottom_size, bottom_ptr);
  ImageBatch top = NArray::MakeNArray(top_size, top_ptr);
  ImageBatch top_diff = NArray::MakeNArray(top_diff_size, top_diff_ptr);

  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 1, 1, 1, 1);
  ImageBatch output = Convolution::PoolingBackward(top_diff, top, bottom, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), bottom_size);
  for (int i = 0; i < bottom_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], expected_raw[i], 0.001);
	//printf("%f, ",output_ptr.get()[i]);
  }
}


TEST(PoolingBackward, CpuWithInsufficientPadding) {
  float bottom_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float top_raw[] = {6, 8, 8, 14, 16, 16, 14, 16, 16};
  float top_diff_raw[] = {0.5,0.4,0.1,0.9,1.2,0.3,1.4,1.5,2.1};
  float expected_raw[] = {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.500000, 0.000000, 0.500000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 2.300000, 0.000000, 5.100000};



  Scale bottom_size{4, 4, 1, 1};
  Scale top_size{3, 3, 1, 1};
  Scale top_diff_size{3, 3, 1, 1};

  shared_ptr<float> bottom_ptr(new float[bottom_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_ptr(new float[top_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_diff_ptr(new float[top_diff_size.Prod()], [](float* ptr) { delete[] ptr; });

  memcpy(bottom_ptr.get(), bottom_raw, bottom_size.Prod() * sizeof(float));
  memcpy(top_ptr.get(), top_raw, top_size.Prod() * sizeof(float));
  memcpy(top_diff_ptr.get(), top_diff_raw, top_diff_size.Prod() * sizeof(float));


  ImageBatch bottom = NArray::MakeNArray(bottom_size, bottom_ptr);
  ImageBatch top = NArray::MakeNArray(top_size, top_ptr);
  ImageBatch top_diff = NArray::MakeNArray(top_diff_size, top_diff_ptr);

  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 2, 2, 1, 1);
  ImageBatch output = Convolution::PoolingBackward(top_diff, top, bottom, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), bottom_size);
  for (int i = 0; i < bottom_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], expected_raw[i], 0.001);
	//printf("%f, ",output_ptr.get()[i]);
  }
}

TEST(PoolingBackward, CpuWithTooMuchPadding) {
  float bottom_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float top_raw[] = {6, 8, 14, 16};
  float top_diff_raw[] = {0.3,1.4,1.5,2.1};
  float expected_raw[] = {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.300000, 0.000000, 1.400000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.500000, 0.000000, 2.100000};


  Scale bottom_size{4, 4, 1, 1};
  Scale top_size{2, 2, 1, 1};
  Scale top_diff_size{2, 2, 1, 1};

  shared_ptr<float> bottom_ptr(new float[bottom_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_ptr(new float[top_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> top_diff_ptr(new float[top_diff_size.Prod()], [](float* ptr) { delete[] ptr; });

  memcpy(bottom_ptr.get(), bottom_raw, bottom_size.Prod() * sizeof(float));
  memcpy(top_ptr.get(), top_raw, top_size.Prod() * sizeof(float));
  memcpy(top_diff_ptr.get(), top_diff_raw, top_diff_size.Prod() * sizeof(float));


  ImageBatch bottom = NArray::MakeNArray(bottom_size, bottom_ptr);
  ImageBatch top = NArray::MakeNArray(top_size, top_ptr);
  ImageBatch top_diff = NArray::MakeNArray(top_diff_size, top_diff_ptr);

  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 4, 4, 3, 3, 2, 2);
  ImageBatch output = Convolution::PoolingBackward(top_diff, top, bottom, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), bottom_size);
  for (int i = 0; i < bottom_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], expected_raw[i], 0.001);
	//printf("%f, ",output_ptr.get()[i]);
  }
}

