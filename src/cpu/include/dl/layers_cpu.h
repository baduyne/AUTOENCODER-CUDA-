#pragma once
#include <torch/torch.h>
#include "dl/conv2d_cpu.h"

namespace dl {

// -------- Conv2D Layer (CPU) --------
class Conv2D {
public:
    Conv2D(int in_channels,
           int out_channels,
           int kernel_size = 3,
           int stride = 1,
           int padding = 1,
           const torch::Device& device = torch::kCPU);

    // forward trên CPU (dùng conv2d_cpu)
    torch::Tensor forward(const torch::Tensor& input) const;

    // getter/setter cho weight/bias (dùng save/load, update)
    const torch::Tensor& weight() const { return weight_; }
    const torch::Tensor& bias()   const { return bias_;  }
    torch::Tensor& weight() { return weight_; }
    torch::Tensor& bias()   { return bias_;  }

    int stride()  const { return stride_;  }
    int padding() const { return padding_; }

private:
    torch::Tensor weight_;    // [C_out, C_in, K, K]
    torch::Tensor bias_;      // [C_out]
    int stride_;
    int padding_;
};

// -------- ReLU (CPU) --------

// Forward: y = max(0, x)
torch::Tensor relu_cpu(const torch::Tensor& input);

// Backward: dL/dx = dL/dy * (y > 0 ? 1 : 0)
// dùng output y của ReLU để làm mask
torch::Tensor relu_backward_cpu(const torch::Tensor& grad_output,
                                const torch::Tensor& relu_output);

// -------- Max Pooling 2x2 (CPU) --------

// Forward: MaxPool2D(2x2, stride=2)
// Input:  [N, C, H, W]
// Output: [N, C, H/2, W/2]
torch::Tensor maxpool2d_2x2_cpu(const torch::Tensor& input);

// Backward:
// grad_output: [N, C, H/2, W/2]
// input:       [N, C, H, W] (giá trị trước pooling)
// grad_input:  [N, C, H, W]
torch::Tensor maxpool2d_2x2_backward_cpu(const torch::Tensor& grad_output,
                                         const torch::Tensor& input);

// -------- Upsampling 2x (nearest neighbor, CPU) --------

// Forward: nearest neighbor 2x
// Input:  [N, C, H, W]
// Output: [N, C, H*2, W*2]
torch::Tensor upsample_nearest2x_cpu(const torch::Tensor& input);

// Backward:
// grad_output: [N, C, H*2, W*2]
// input:       [N, C, H, W] (giá trị trước upsample)
// grad_input:  [N, C, H, W]
torch::Tensor upsample_nearest2x_backward_cpu(const torch::Tensor& grad_output,
                                              const torch::Tensor& input);

// -------- MSE Loss (CPU) --------
// L = mean((output - target)^2)
torch::Tensor mse_loss_cpu(const torch::Tensor& output,
                           const torch::Tensor& target);

// Backward cho MSE: trả về dL/dOutput
// dL/dOutput = 2 * (output - target) / numel
torch::Tensor mse_loss_backward_cpu(const torch::Tensor& output,
                                    const torch::Tensor& target);

} // namespace dl
