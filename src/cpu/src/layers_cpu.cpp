#include "dl/layers_cpu.h"

#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace dl {

// ===================== Conv2D Layer =====================

Conv2D::Conv2D(int in_channels,
               int out_channels,
               int kernel_size,
               int stride,
               int padding,
               const torch::Device& device)
    : stride_(stride)
    , padding_(padding)
{
    if (kernel_size != 3) {
        throw std::runtime_error("Conv2D: hiện tại chỉ hỗ trợ kernel_size = 3");
    }

    // Kaiming (He) normal initialization suitable for ReLU activations.
    auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
    const double fan_in = static_cast<double>(in_channels * kernel_size * kernel_size);
    const double std = std::sqrt(2.0 / std::max(1.0, fan_in));

    weight_ = torch::randn(
        {out_channels, in_channels, kernel_size, kernel_size},
        options
    ) * static_cast<float>(std);

    bias_ = torch::zeros({out_channels}, options);
}

torch::Tensor Conv2D::forward(const torch::Tensor& input) const {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("Conv2D::forward (CPU): input must be on CPU");
    }
    if (!weight_.device().is_cpu() || !bias_.device().is_cpu()) {
        throw std::runtime_error("Conv2D::forward (CPU): weights/bias must be on CPU");
    }

    // Dùng PyTorch built-in conv2d (tối ưu với MKL/OpenBLAS)
    return torch::nn::functional::conv2d(
        input,
        weight_,
        torch::nn::functional::Conv2dFuncOptions()
            .bias(bias_)
            .stride(stride_)
            .padding(padding_)
    );
}

// ===================== ReLU (CPU) =====================

torch::Tensor relu_cpu(const torch::Tensor& input) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("relu_cpu: input must be on CPU");
    }
    if (input.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("relu_cpu: only float32 supported");
    }

    auto x = input.contiguous();
    auto y = torch::empty_like(x);

    const auto numel = x.numel();
    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr       = y.data_ptr<float>();

    for (int64_t i = 0; i < numel; ++i) {
        float v = x_ptr[i];
        y_ptr[i] = v > 0.0f ? v : 0.0f;
    }

    return y;
}

torch::Tensor relu_backward_cpu(const torch::Tensor& grad_output,
                                const torch::Tensor& relu_output) {
    if (!grad_output.device().is_cpu() || !relu_output.device().is_cpu()) {
        throw std::runtime_error("relu_backward_cpu: tensors must be on CPU");
    }
    auto go = grad_output.contiguous();
    auto y  = relu_output.contiguous();

    if (!go.sizes().equals(y.sizes())) {
        throw std::runtime_error("relu_backward_cpu: grad_output and relu_output size mismatch");
    }

    torch::Tensor grad_input = torch::empty_like(go);

    const float* go_ptr = go.data_ptr<float>();
    const float* y_ptr  = y.data_ptr<float>();
    float* gi_ptr       = grad_input.data_ptr<float>();

    const auto numel = go.numel();
    for (int64_t i = 0; i < numel; ++i) {
        // derivative=1 if y>0, else 0
        gi_ptr[i] = (y_ptr[i] > 0.0f) ? go_ptr[i] : 0.0f;
    }

    return grad_input;
}

// ===================== MaxPool 2x2 (CPU) =====================

torch::Tensor maxpool2d_2x2_cpu(const torch::Tensor& input) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("maxpool2d_2x2_cpu: input must be on CPU");
    }
    if (input.dim() != 4) {
        throw std::runtime_error("maxpool2d_2x2_cpu: input must be 4D [N, C, H, W]");
    }

    auto x = input.contiguous();
    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t H_out = H / 2;
    const int64_t W_out = W / 2;

    torch::Tensor output = torch::empty(
        {N, C, H_out, W_out},
        x.options()
    );

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr       = output.data_ptr<float>();

    auto idx_in = [=](int64_t n, int64_t c, int64_t h, int64_t w_) {
        return ((n * C + c) * H + h) * W + w_;
    };
    auto idx_out = [=](int64_t n, int64_t c, int64_t h, int64_t w_) {
        return ((n * C + c) * H_out + h) * W_out + w_;
    };

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t h0 = oh * 2;
                    int64_t w0 = ow * 2;

                    float m = x_ptr[idx_in(n, c, h0,     w0    )];
                    m = std::max(m, x_ptr[idx_in(n, c, h0,     w0 + 1)]);
                    m = std::max(m, x_ptr[idx_in(n, c, h0 + 1, w0    )]);
                    m = std::max(m, x_ptr[idx_in(n, c, h0 + 1, w0 + 1)]);

                    y_ptr[idx_out(n, c, oh, ow)] = m;
                }
            }
        }
    }

    return output;
}

torch::Tensor maxpool2d_2x2_backward_cpu(const torch::Tensor& grad_output,
                                         const torch::Tensor& input) {
    if (!grad_output.device().is_cpu() || !input.device().is_cpu()) {
        throw std::runtime_error("maxpool2d_2x2_backward_cpu: tensors must be on CPU");
    }
    auto go = grad_output.contiguous();
    auto x  = input.contiguous();

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t H_out = go.size(2);
    const int64_t W_out = go.size(3);

    if (H_out * 2 != H || W_out * 2 != W) {
        throw std::runtime_error("maxpool2d_2x2_backward_cpu: shape mismatch");
    }

    torch::Tensor grad_input = torch::zeros_like(x);

    const float* x_ptr  = x.data_ptr<float>();
    const float* go_ptr = go.data_ptr<float>();
    float* gi_ptr       = grad_input.data_ptr<float>();

    auto idx_in = [=](int64_t n, int64_t c, int64_t h, int64_t w_) {
        return ((n * C + c) * H + h) * W + w_;
    };
    auto idx_out = [=](int64_t n, int64_t c, int64_t h, int64_t w_) {
        return ((n * C + c) * H_out + h) * W_out + w_;
    };

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t h0 = oh * 2;
                    int64_t w0 = ow * 2;

                    // Tìm vị trí max trong 4 ô
                    float v00 = x_ptr[idx_in(n, c, h0,     w0    )];
                    float v01 = x_ptr[idx_in(n, c, h0,     w0 + 1)];
                    float v10 = x_ptr[idx_in(n, c, h0 + 1, w0    )];
                    float v11 = x_ptr[idx_in(n, c, h0 + 1, w0 + 1)];

                    float m = v00;
                    int64_t mh = h0;
                    int64_t mw = w0;

                    if (v01 > m) { m = v01; mh = h0;     mw = w0 + 1; }
                    if (v10 > m) { m = v10; mh = h0 + 1; mw = w0;     }
                    if (v11 > m) { m = v11; mh = h0 + 1; mw = w0 + 1; }

                    float g = go_ptr[idx_out(n, c, oh, ow)];
                    gi_ptr[idx_in(n, c, mh, mw)] += g;
                }
            }
        }
    }

    return grad_input;
}

// ===================== Upsampling 2x (nearest, CPU) =====================

torch::Tensor upsample_nearest2x_cpu(const torch::Tensor& input) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("upsample_nearest2x_cpu: input must be on CPU");
    }
    if (input.dim() != 4) {
        throw std::runtime_error("upsample_nearest2x_cpu: input must be 4D [N, C, H, W]");
    }

    auto x = input.contiguous();
    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t H_out = H * 2;
    const int64_t W_out = W * 2;

    torch::Tensor output = torch::empty(
        {N, C, H_out, W_out},
        x.options()
    );

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr       = output.data_ptr<float>();

    auto idx_in = [=](int64_t n, int64_t c, int64_t h, int64_t w_) {
        return ((n * C + c) * H + h) * W + w_;
    };
    auto idx_out = [=](int64_t n, int64_t c, int64_t h, int64_t w_) {
        return ((n * C + c) * H_out + h) * W_out + w_;
    };

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t ih = oh / 2;
                    int64_t iw = ow / 2;

                    float v = x_ptr[idx_in(n, c, ih, iw)];
                    y_ptr[idx_out(n, c, oh, ow)] = v;
                }
            }
        }
    }

    return output;
}

torch::Tensor upsample_nearest2x_backward_cpu(const torch::Tensor& grad_output,
                                              const torch::Tensor& input) {
    if (!grad_output.device().is_cpu() || !input.device().is_cpu()) {
        throw std::runtime_error("upsample_nearest2x_backward_cpu: tensors must be on CPU");
    }
    auto go = grad_output.contiguous();
    auto x  = input.contiguous();

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t H_out = go.size(2);
    const int64_t W_out = go.size(3);

    if (H_out != H * 2 || W_out != W * 2) {
        throw std::runtime_error("upsample_nearest2x_backward_cpu: shape mismatch");
    }

    torch::Tensor grad_input = torch::zeros_like(x);

    const float* go_ptr = go.data_ptr<float>();
    float* gi_ptr       = grad_input.data_ptr<float>();

    auto idx_in = [=](int64_t n, int64_t c, int64_t h, int64_t w_) {
        return ((n * C + c) * H + h) * W + w_;
    };
    auto idx_out = [=](int64_t n, int64_t c, int64_t h, int64_t w_) {
        return ((n * C + c) * H_out + h) * W_out + w_;
    };

    // Mỗi pixel input nhận grad từ 4 pixel output tương ứng
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ih = 0; ih < H; ++ih) {
                for (int64_t iw = 0; iw < W; ++iw) {
                    float sum = 0.0f;
                    int64_t oh0 = ih * 2;
                    int64_t ow0 = iw * 2;

                    sum += go_ptr[idx_out(n, c, oh0,     ow0    )];
                    sum += go_ptr[idx_out(n, c, oh0,     ow0 + 1)];
                    sum += go_ptr[idx_out(n, c, oh0 + 1, ow0    )];
                    sum += go_ptr[idx_out(n, c, oh0 + 1, ow0 + 1)];

                    gi_ptr[idx_in(n, c, ih, iw)] = sum;
                }
            }
        }
    }

    return grad_input;
}

// ===================== MSE Loss (CPU) =====================

torch::Tensor mse_loss_cpu(const torch::Tensor& output,
                           const torch::Tensor& target) {
    if (!output.device().is_cpu() || !target.device().is_cpu()) {
        throw std::runtime_error("mse_loss_cpu: tensors must be on CPU");
    }
    if (!output.sizes().equals(target.sizes())) {
        throw std::runtime_error("mse_loss_cpu: output and target must have same shape");
    }

    torch::Tensor diff = output - target;
    torch::Tensor sq   = diff.mul(diff);
    torch::Tensor loss = sq.mean();
    return loss;
}

torch::Tensor mse_loss_backward_cpu(const torch::Tensor& output,
                                    const torch::Tensor& target) {
    if (!output.device().is_cpu() || !target.device().is_cpu()) {
        throw std::runtime_error("mse_loss_backward_cpu: tensors must be on CPU");
    }
    if (!output.sizes().equals(target.sizes())) {
        throw std::runtime_error("mse_loss_backward_cpu: output and target must have same shape");
    }

    auto diff = (output - target).contiguous();
    torch::Tensor grad = torch::empty_like(diff);

    float* g_ptr      = grad.data_ptr<float>();
    const float* d_ptr = diff.data_ptr<float>();

    const auto numel = diff.numel();
    float scale = 2.0f / static_cast<float>(numel);
    for (int64_t i = 0; i < numel; ++i) {
        g_ptr[i] = scale * d_ptr[i]; // dL/dOutput
    }

    return grad;
}

} // namespace dl
