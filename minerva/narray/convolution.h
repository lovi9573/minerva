#pragma once
#include "narray/image_batch.h"
#include "narray/convolution_info.h"

namespace minerva {

class Convolution {
 public:
  static ImageBatch ConvForward(ImageBatch src, Filter filter, NArray bias, ConvInfo info);
  static ImageBatch ConvBackwardData(ImageBatch diff, ImageBatch bottom, Filter filter, ConvInfo info);
  static Filter ConvBackwardFilter(ImageBatch diff, ImageBatch bottom, Filter filter, ConvInfo info);
  static NArray ConvBackwardBias(ImageBatch diff);
  static ImageBatch SoftmaxForward(ImageBatch src, SoftmaxAlgorithm algorithm);
  static ImageBatch SoftmaxBackward(ImageBatch diff, ImageBatch top, SoftmaxAlgorithm algorithm);
  static ImageBatch ActivationForward(ImageBatch src, ActivationAlgorithm algorithm);
  static ImageBatch ActivationBackward(ImageBatch diff, ImageBatch top, ImageBatch bottom, ActivationAlgorithm algorithm);
  static ImageBatch PoolingForward(ImageBatch src, PoolingInfo info);
  static ImageBatch PoolingBackward(ImageBatch diff, ImageBatch top, ImageBatch bottom, PoolingInfo info);

  static ImageBatch LRNForward(ImageBatch src, ImageBatch scale, int local_size, element_t alpha, element_t beta);
  static ImageBatch LRNBackward(ImageBatch bottom_data, ImageBatch top_data, ImageBatch scale, ImageBatch top_diff , int local_size, element_t alpha, element_t beta);

};

}  // namespace minerva

