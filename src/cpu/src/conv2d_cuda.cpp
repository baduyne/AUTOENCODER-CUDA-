#include <torch/torch.h>
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include "dl/conv2d_cuda.h"

namespace dl {

__global__ void conv2d_naive_kernel(
    const float* __restrict__ x,    // (N, C_in, H, W)
    const float* __restrict__ w,    // (C_out, C_in, K, K)
    const float* __restrict__ b,    // (C_out)
    float* __restrict__ y,          // (N, C_out, H_out, W_out)
    int N, int C_in, int H, int W,
    int C_out, int K,
    int H_out, int W_out,
    int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    // decode idx -> (n, co, oh, ow)
    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp /= H_out;
    int co = tmp % C_out;
    int n  = tmp / C_out;

    auto idx_in = [=] __device__ (int n_, int c_, int h_, int w_) {
        return ((n_ * C_in + c_) * H + h_) * W + w_;
    };
    auto idx_w = [=] __device__ (int co_, int ci_, int kh_, int kw_) {
        return ((co_ * C_in + ci_) * K + kh_) * K + kw_;
    };
    auto idx_out = [=] __device__ (int n_, int co_, int h_, int w_) {
        return ((n_ * C_out + co_) * H_out + h_) * W_out + w_;
    };

    float sum = b[co];

    int ih_center = oh * stride - padding;
    int iw_center = ow * stride - padding;

    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = ih_center + kh;
                int iw = iw_center + kw;

                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                    continue;
                }
                int in_i = idx_in(n, ci, ih, iw);
                int w_i  = idx_w(co, ci, kh, kw);

                sum += x[in_i] * w[w_i];
            }
        }
    }

    int out_i = idx_out(n, co, oh, ow);
    y[out_i] = sum;
}

torch::Tensor conv2d_cuda(const torch::Tensor& input,
                          const torch::Tensor& weight,
                          const torch::Tensor& bias,
                          int stride,
                          int padding) {
    TORCH_CHECK(input.is_cuda(), "conv2d_cuda: input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv2d_cuda: weight must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "conv2d_cuda: bias must be CUDA tensor");

    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto b = bias.contiguous();

    const int N     = x.size(0);
    const int C_in  = x.size(1);
    const int H     = x.size(2);
    const int W     = x.size(3);

    const int C_out = w.size(0);
    const int K     = w.size(2);

    const int H_out = (H + 2 * padding - K) / stride + 1;
    const int W_out = (W + 2 * padding - K) / stride + 1;

    auto y = torch::zeros({N, C_out, H_out, W_out}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* y_ptr       = y.data_ptr<float>();

    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_naive_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, b_ptr, y_ptr,
        N, C_in, H, W,
        C_out, K,
        H_out, W_out,
        stride, padding
    );
    cudaDeviceSynchronize();

    return y;
}

} // namespace dl

#endif // USE_CUDA
