#include "dl/conv2d_cpu.h"
#include <cassert>
#include <stdexcept>

namespace dl {

// ===================== FORWARD =====================

torch::Tensor conv2d_cpu(const torch::Tensor& input,
                         const torch::Tensor& weight,
                         const torch::Tensor& bias,
                         int stride,
                         int padding) {
    // Đảm bảo float32 & 4D
    assert(input.dim() == 4);
    assert(weight.dim() == 4);
    assert(bias.dim() == 1);

    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto b = bias.contiguous();

    const int64_t N     = x.size(0);
    const int64_t C_in  = x.size(1);
    const int64_t H     = x.size(2);
    const int64_t W     = x.size(3);

    const int64_t C_out = w.size(0);
    const int64_t K     = w.size(2); // giả định KxK

    const int64_t H_out = (H + 2 * padding - K) / stride + 1;
    const int64_t W_out = (W + 2 * padding - K) / stride + 1;

    torch::Tensor output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* y_ptr       = output.data_ptr<float>();

    auto idx_in = [=](int64_t n, int64_t c, int64_t h, int64_t w_) {
        return ((n * C_in + c) * H + h) * W + w_;
    };
    auto idx_w = [=](int64_t co, int64_t ci, int64_t kh, int64_t kw) {
        return ((co * C_in + ci) * K + kh) * K + kw;
    };
    auto idx_out = [=](int64_t n, int64_t co, int64_t h, int64_t w_) {
        return ((n * C_out + co) * H_out + h) * W_out + w_;
    };

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t co = 0; co < C_out; ++co) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {

                    float sum = b_ptr[co];

                    const int64_t ih_center = oh * stride - padding;
                    const int64_t iw_center = ow * stride - padding;

                    for (int64_t ci = 0; ci < C_in; ++ci) {
                        for (int64_t kh = 0; kh < K; ++kh) {
                            for (int64_t kw = 0; kw < K; ++kw) {
                                int64_t ih = ih_center + kh;
                                int64_t iw = iw_center + kw;

                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                    continue;
                                }

                                int64_t in_i = idx_in(n, ci, ih, iw);
                                int64_t w_i  = idx_w(co, ci, kh, kw);

                                sum += x_ptr[in_i] * w_ptr[w_i];
                            }
                        }
                    }

                    int64_t out_i = idx_out(n, co, oh, ow);
                    y_ptr[out_i] = sum;
                }
            }
        }
    }

    return output;
}

// ===================== BACKWARD =====================

Conv2DGrad conv2d_backward_cpu(const torch::Tensor& input,
                               const torch::Tensor& weight,
                               const torch::Tensor& grad_output,
                               int stride,
                               int padding) {
    auto x   = input.contiguous();
    auto w   = weight.contiguous();
    auto gy  = grad_output.contiguous();

    const int64_t N     = x.size(0);
    const int64_t C_in  = x.size(1);
    const int64_t H     = x.size(2);
    const int64_t W     = x.size(3);

    const int64_t C_out = w.size(0);
    const int64_t K     = w.size(2);

    const int64_t H_out = gy.size(2);
    const int64_t W_out = gy.size(3);

    Conv2DGrad grads;
    grads.grad_input  = torch::zeros_like(x);       // [N, C_in, H, W]
    grads.grad_weight = torch::zeros_like(w);       // [C_out, C_in, K, K]
    grads.grad_bias   = torch::zeros({C_out},      // [C_out]
                                     w.options());

    const float* x_ptr    = x.data_ptr<float>();
    const float* w_ptr    = w.data_ptr<float>();
    const float* gy_ptr   = gy.data_ptr<float>();
    float* gx_ptr         = grads.grad_input.data_ptr<float>();
    float* gw_ptr         = grads.grad_weight.data_ptr<float>();
    float* gb_ptr         = grads.grad_bias.data_ptr<float>();

    auto idx_in = [=](int64_t n, int64_t c, int64_t h, int64_t w_) {
        return ((n * C_in + c) * H + h) * W + w_;
    };
    auto idx_w = [=](int64_t co, int64_t ci, int64_t kh, int64_t kw) {
        return ((co * C_in + ci) * K + kh) * K + kw;
    };
    auto idx_out = [=](int64_t n, int64_t co, int64_t h, int64_t w_) {
        return ((n * C_out + co) * H_out + h) * W_out + w_;
    };

    // Loop theo output, phân phối gradient về input + weight + bias
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t co = 0; co < C_out; ++co) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {

                    int64_t out_i = idx_out(n, co, oh, ow);
                    float g = gy_ptr[out_i]; // dL/dY[n,co,oh,ow]

                    // bias grad
                    gb_ptr[co] += g;

                    const int64_t ih_center = oh * stride - padding;
                    const int64_t iw_center = ow * stride - padding;

                    for (int64_t ci = 0; ci < C_in; ++ci) {
                        for (int64_t kh = 0; kh < K; ++kh) {
                            for (int64_t kw = 0; kw < K; ++kw) {
                                int64_t ih = ih_center + kh;
                                int64_t iw = iw_center + kw;

                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                    continue;
                                }

                                int64_t in_i = idx_in(n, ci, ih, iw);
                                int64_t w_i  = idx_w(co, ci, kh, kw);

                                float x_val = x_ptr[in_i];
                                float w_val = w_ptr[w_i];

                                // grad w
                                gw_ptr[w_i] += g * x_val;

                                // grad x
                                gx_ptr[in_i] += g * w_val;
                            }
                        }
                    }
                }
            }
        }
    }

    return grads;
}

} // namespace dl
