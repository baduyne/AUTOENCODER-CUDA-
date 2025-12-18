#pragma once
#include <torch/torch.h>

namespace dl {

// Chọn device mặc định dựa vào USE_CUDA và tình trạng GPU runtime
torch::Device get_default_device();

// GPU có sẵn không (chỉ meaningful khi build USE_CUDA=ON)
bool cuda_available();

} // namespace dl
