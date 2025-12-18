#pragma once
#include <torch/torch.h>

namespace dl {

// Forward conv2d CPU (naive)
// input:  [N, C_in, H, W]
// weight: [C_out, C_in, K, K]
// bias:   [C_out]
// output: [N, C_out, H_out, W_out]
torch::Tensor conv2d_cpu(const torch::Tensor& input,
                         const torch::Tensor& weight,
                         const torch::Tensor& bias,
                         int stride = 1,
                         int padding = 1);

// Struct chứa gradient cho conv2d
struct Conv2DGrad {
    torch::Tensor grad_input;   // [N, C_in, H, W]
    torch::Tensor grad_weight;  // [C_out, C_in, K, K]
    torch::Tensor grad_bias;    // [C_out]
};

// Backward conv2d CPU
// Cho input, weight, grad_output -> trả về grad_input, grad_weight, grad_bias
Conv2DGrad conv2d_backward_cpu(const torch::Tensor& input,
                               const torch::Tensor& weight,
                               const torch::Tensor& grad_output,
                               int stride = 1,
                               int padding = 1);

} // namespace dl
