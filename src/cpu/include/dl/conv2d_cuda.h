#pragma once
#include <torch/torch.h>

#ifdef USE_CUDA

namespace dl {

// Naive conv2d GPU: interface giống conv2d_cpu
torch::Tensor conv2d_cuda(const torch::Tensor& input,
                          const torch::Tensor& weight,
                          const torch::Tensor& bias,
                          int stride = 1,
                          int padding = 1);

} // namespace dl

#endif // USE_CUDA
