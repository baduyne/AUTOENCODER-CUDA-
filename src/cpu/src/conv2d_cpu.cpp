#include "dl/conv2d_cpu.h"
#include <cstring>
#include <algorithm>

namespace dl {

// Utility: index flatten for [N, C, H, W]
inline int idx4(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}
// Utility: index flatten for [Cout, Cin, K, K]
inline int idx4w(int co, int ci, int kh, int kw, int Cin, int K) {
    return ((co * Cin + ci) * K + kh) * K + kw;
}

void conv2d_forward_cpu(const float* input,
                        const float* weight,
                        const float* bias,
                        float* output,
                        int N, int Cin, int H, int W,
                        int Cout, int K, int stride, int padding) {
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    std::fill(output, output + N * Cout * H_out * W_out, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < Cout; ++co) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = bias ? bias[co] : 0.f;
                    int ih_center = oh * stride - padding;
                    int iw_center = ow * stride - padding;
                    for (int ci = 0; ci < Cin; ++ci) {
                        for (int kh = 0; kh < K; ++kh) {
                            for (int kw = 0; kw < K; ++kw) {
                                int ih = ih_center + kh;
                                int iw = iw_center + kw;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                int in_idx = idx4(n, ci, ih, iw, Cin, H, W);
                                int w_idx  = idx4w(co, ci, kh, kw, Cin, K);
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    int out_idx = idx4(n, co, oh, ow, Cout, H_out, W_out);
                    output[out_idx] = sum;
                }
            }
        }
    }
}

void conv2d_backward_cpu(const float* input,
                         const float* weight,
                         const float* grad_output,
                         float* grad_input,
                         float* grad_weight,
                         float* grad_bias,
                         int N, int Cin, int H, int W,
                         int Cout, int K, int stride, int padding) {
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    std::fill(grad_input, grad_input + N * Cin * H * W, 0.0f);
    std::fill(grad_weight, grad_weight + Cout * Cin * K * K, 0.0f);
    std::fill(grad_bias, grad_bias + Cout, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < Cout; ++co) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    int out_idx = idx4(n, co, oh, ow, Cout, H_out, W_out);
                    float go = grad_output[out_idx];
                        grad_bias[co] += go; // Summing gradients for bias
                        int ih_center = oh * stride - padding; // Input height center
                        int iw_center = ow * stride - padding; // Input width center
                    for (int ci = 0; ci < Cin; ++ci) {
                        for (int kh = 0; kh < K; ++kh) {
                            for (int kw = 0; kw < K; ++kw) {
                                int ih = ih_center + kh;
                                int iw = iw_center + kw;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                int in_idx = idx4(n, ci, ih, iw, Cin, H, W);
                                int w_idx  = idx4w(co, ci, kh, kw, Cin, K);
                                // grad weight
                                    grad_weight[w_idx] += input[in_idx] * go; // Summing gradients for weight
                                // grad input
                                grad_input[in_idx] += weight[w_idx] * go;
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace dl
