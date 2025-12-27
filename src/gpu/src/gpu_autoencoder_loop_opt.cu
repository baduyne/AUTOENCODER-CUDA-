#include "gpu_autoencoder_loop_opt.h"
#include <algorithm>

// Macro kiểm tra lỗi CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Tiling parameters
#define TILE_W 16
#define TILE_H 16

// Index helper for NCHW layout
__host__ __device__ inline int idx4_opt(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

// ============================================================================
// OPTIMIZED CONVOLUTION KERNELS (3x3 FULLY UNROLLED)
// ============================================================================

__global__ void conv2d_forward_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    // 3x3 convolution with padding=1, stride=1
    const int pad = 1;
    const int stride = 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * height * width;
    if (tid >= total_outputs) return;

    // Decode indices: (n, oc, oh, ow)
    int tmp = tid;
    int ow = tmp % width;          tmp /= width;
    int oh = tmp % height;         tmp /= height;
    int oc = tmp % out_channels;   tmp /= out_channels;
    int n  = tmp;

    float acc = 0.0f;

    // Loop over input channels (can be pragma unrolled if in_channels is small)
    for (int ic = 0; ic < in_channels; ++ic) {
        // FULLY UNROLLED 3x3 KERNEL
        // ky=0, kx=0
        {
            int iy = oh * stride + 0 - pad;
            int ix = ow * stride + 0 - pad;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int in_idx = idx4_opt(n, ic, iy, ix, in_channels, height, width);
                int w_idx = ((oc * in_channels + ic) * 3 + 0) * 3 + 0;
                acc += input[in_idx] * weights[w_idx];
            }
        }
        // ky=0, kx=1
        {
            int iy = oh * stride + 0 - pad;
            int ix = ow * stride + 1 - pad;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int in_idx = idx4_opt(n, ic, iy, ix, in_channels, height, width);
                int w_idx = ((oc * in_channels + ic) * 3 + 0) * 3 + 1;
                acc += input[in_idx] * weights[w_idx];
            }
        }
        // ky=0, kx=2
        {
            int iy = oh * stride + 0 - pad;
            int ix = ow * stride + 2 - pad;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int in_idx = idx4_opt(n, ic, iy, ix, in_channels, height, width);
                int w_idx = ((oc * in_channels + ic) * 3 + 0) * 3 + 2;
                acc += input[in_idx] * weights[w_idx];
            }
        }
        // ky=1, kx=0
        {
            int iy = oh * stride + 1 - pad;
            int ix = ow * stride + 0 - pad;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int in_idx = idx4_opt(n, ic, iy, ix, in_channels, height, width);
                int w_idx = ((oc * in_channels + ic) * 3 + 1) * 3 + 0;
                acc += input[in_idx] * weights[w_idx];
            }
        }
        // ky=1, kx=1
        {
            int iy = oh * stride + 1 - pad;
            int ix = ow * stride + 1 - pad;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int in_idx = idx4_opt(n, ic, iy, ix, in_channels, height, width);
                int w_idx = ((oc * in_channels + ic) * 3 + 1) * 3 + 1;
                acc += input[in_idx] * weights[w_idx];
            }
        }
        // ky=1, kx=2
        {
            int iy = oh * stride + 1 - pad;
            int ix = ow * stride + 2 - pad;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int in_idx = idx4_opt(n, ic, iy, ix, in_channels, height, width);
                int w_idx = ((oc * in_channels + ic) * 3 + 1) * 3 + 2;
                acc += input[in_idx] * weights[w_idx];
            }
        }
        // ky=2, kx=0
        {
            int iy = oh * stride + 2 - pad;
            int ix = ow * stride + 0 - pad;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int in_idx = idx4_opt(n, ic, iy, ix, in_channels, height, width);
                int w_idx = ((oc * in_channels + ic) * 3 + 2) * 3 + 0;
                acc += input[in_idx] * weights[w_idx];
            }
        }
        // ky=2, kx=1
        {
            int iy = oh * stride + 2 - pad;
            int ix = ow * stride + 1 - pad;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int in_idx = idx4_opt(n, ic, iy, ix, in_channels, height, width);
                int w_idx = ((oc * in_channels + ic) * 3 + 2) * 3 + 1;
                acc += input[in_idx] * weights[w_idx];
            }
        }
        // ky=2, kx=2
        {
            int iy = oh * stride + 2 - pad;
            int ix = ow * stride + 2 - pad;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int in_idx = idx4_opt(n, ic, iy, ix, in_channels, height, width);
                int w_idx = ((oc * in_channels + ic) * 3 + 2) * 3 + 2;
                acc += input[in_idx] * weights[w_idx];
            }
        }
    }

    int out_idx = idx4_opt(n, oc, oh, ow, out_channels, height, width);
    output[out_idx] = acc + bias[oc];
}

void gpu_conv2d_forward_opt(
    const float* dev_input_data, const float* d_weights, const float* d_bias,
    float* d_output, int batch_size, int in_channels, int out_channels,
    int height, int width
) {
    int total_outputs = batch_size * out_channels * height * width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    conv2d_forward_kernel_opt<<<grid_size, block_size>>>(
        dev_input_data, d_weights, d_bias, d_output,
        batch_size, in_channels, out_channels, height, width
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void conv2d_backward_input_kernel_opt(
    const float* __restrict__ weights,
    const float* __restrict__ dL_doutput,
    float* __restrict__ dL_dinput,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    const int pad = 1;
    const int stride = 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * in_channels * height * width;
    if (tid >= total) return;

    int tmp = tid;
    int iw = tmp % width;     tmp /= width;
    int ih = tmp % height;    tmp /= height;
    int ic = tmp % in_channels; tmp /= in_channels;
    int n  = tmp;

    float acc = 0.0f;

    for (int oc = 0; oc < out_channels; ++oc) {
        // FULLY UNROLLED 3x3 KERNEL
        // ky=0, kx=0
        {
            int oh = ih + pad - 0;
            int ow = iw + pad - 0;
            if (oh % stride == 0 && ow % stride == 0) {
                oh /= stride;
                ow /= stride;
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int dout_idx = idx4_opt(n, oc, oh, ow, out_channels, height, width);
                    int w_idx = ((oc * in_channels + ic) * 3 + 0) * 3 + 0;
                    acc += dL_doutput[dout_idx] * weights[w_idx];
                }
            }
        }
        // ky=0, kx=1
        {
            int oh = ih + pad - 0;
            int ow = iw + pad - 1;
            if (oh % stride == 0 && ow % stride == 0) {
                oh /= stride;
                ow /= stride;
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int dout_idx = idx4_opt(n, oc, oh, ow, out_channels, height, width);
                    int w_idx = ((oc * in_channels + ic) * 3 + 0) * 3 + 1;
                    acc += dL_doutput[dout_idx] * weights[w_idx];
                }
            }
        }
        // ky=0, kx=2
        {
            int oh = ih + pad - 0;
            int ow = iw + pad - 2;
            if (oh % stride == 0 && ow % stride == 0) {
                oh /= stride;
                ow /= stride;
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int dout_idx = idx4_opt(n, oc, oh, ow, out_channels, height, width);
                    int w_idx = ((oc * in_channels + ic) * 3 + 0) * 3 + 2;
                    acc += dL_doutput[dout_idx] * weights[w_idx];
                }
            }
        }
        // ky=1, kx=0
        {
            int oh = ih + pad - 1;
            int ow = iw + pad - 0;
            if (oh % stride == 0 && ow % stride == 0) {
                oh /= stride;
                ow /= stride;
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int dout_idx = idx4_opt(n, oc, oh, ow, out_channels, height, width);
                    int w_idx = ((oc * in_channels + ic) * 3 + 1) * 3 + 0;
                    acc += dL_doutput[dout_idx] * weights[w_idx];
                }
            }
        }
        // ky=1, kx=1
        {
            int oh = ih + pad - 1;
            int ow = iw + pad - 1;
            if (oh % stride == 0 && ow % stride == 0) {
                oh /= stride;
                ow /= stride;
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int dout_idx = idx4_opt(n, oc, oh, ow, out_channels, height, width);
                    int w_idx = ((oc * in_channels + ic) * 3 + 1) * 3 + 1;
                    acc += dL_doutput[dout_idx] * weights[w_idx];
                }
            }
        }
        // ky=1, kx=2
        {
            int oh = ih + pad - 1;
            int ow = iw + pad - 2;
            if (oh % stride == 0 && ow % stride == 0) {
                oh /= stride;
                ow /= stride;
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int dout_idx = idx4_opt(n, oc, oh, ow, out_channels, height, width);
                    int w_idx = ((oc * in_channels + ic) * 3 + 1) * 3 + 2;
                    acc += dL_doutput[dout_idx] * weights[w_idx];
                }
            }
        }
        // ky=2, kx=0
        {
            int oh = ih + pad - 2;
            int ow = iw + pad - 0;
            if (oh % stride == 0 && ow % stride == 0) {
                oh /= stride;
                ow /= stride;
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int dout_idx = idx4_opt(n, oc, oh, ow, out_channels, height, width);
                    int w_idx = ((oc * in_channels + ic) * 3 + 2) * 3 + 0;
                    acc += dL_doutput[dout_idx] * weights[w_idx];
                }
            }
        }
        // ky=2, kx=1
        {
            int oh = ih + pad - 2;
            int ow = iw + pad - 1;
            if (oh % stride == 0 && ow % stride == 0) {
                oh /= stride;
                ow /= stride;
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int dout_idx = idx4_opt(n, oc, oh, ow, out_channels, height, width);
                    int w_idx = ((oc * in_channels + ic) * 3 + 2) * 3 + 1;
                    acc += dL_doutput[dout_idx] * weights[w_idx];
                }
            }
        }
        // ky=2, kx=2
        {
            int oh = ih + pad - 2;
            int ow = iw + pad - 2;
            if (oh % stride == 0 && ow % stride == 0) {
                oh /= stride;
                ow /= stride;
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int dout_idx = idx4_opt(n, oc, oh, ow, out_channels, height, width);
                    int w_idx = ((oc * in_channels + ic) * 3 + 2) * 3 + 2;
                    acc += dL_doutput[dout_idx] * weights[w_idx];
                }
            }
        }
    }

    dL_dinput[tid] = acc;
}

__global__ void conv2d_backward_weights_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ dL_doutput,
    float* __restrict__ dL_dweights,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    const int pad = 1;
    const int stride = 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_channels * in_channels * 3 * 3;
    if (tid >= total) return;

    int tmp = tid;
    int kx = tmp % 3; tmp /= 3;
    int ky = tmp % 3; tmp /= 3;
    int ic = tmp % in_channels; tmp /= in_channels;
    int oc = tmp;

    float acc = 0.0f;

    // Partial unroll with pragma
    #pragma unroll 4
    for (int n = 0; n < batch_size; ++n) {
        for (int oh = 0; oh < height; ++oh) {
            for (int ow = 0; ow < width; ++ow) {
                int iy = oh * stride + ky - pad;
                int ix = ow * stride + kx - pad;
                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    int in_idx = idx4_opt(n, ic, iy, ix, in_channels, height, width);
                    int dout_idx = idx4_opt(n, oc, oh, ow, out_channels, height, width);
                    acc += input[in_idx] * dL_doutput[dout_idx];
                }
            }
        }
    }

    int w_idx = ((oc * in_channels + ic) * 3 + ky) * 3 + kx;
    dL_dweights[w_idx] = acc;
}

__global__ void conv2d_backward_bias_kernel_opt(
    const float* __restrict__ dL_doutput,
    float* __restrict__ dL_dbias,
    int batch_size,
    int out_channels,
    int height,
    int width
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= out_channels) return;

    float acc = 0.0f;

    // Partial unroll
    #pragma unroll 4
    for (int n = 0; n < batch_size; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = idx4_opt(n, oc, h, w, out_channels, height, width);
                acc += dL_doutput[idx];
            }
        }
    }
    dL_dbias[oc] = acc;
}

void gpu_conv2d_backward_opt(
    const float* dev_input_data, const float* d_weights, const float* d_dL_doutput,
    float* dev_grad_input, float* d_dL_dweights, float* d_dL_dbias,
    int batch_size, int in_channels, int out_channels, int height, int width
) {
    int block_size = 256;

    // dL/dinput
    if (dev_grad_input) {
        int total_inputs = batch_size * in_channels * height * width;
        int grid_size_input = (total_inputs + block_size - 1) / block_size;
        conv2d_backward_input_kernel_opt<<<grid_size_input, block_size>>>(
            d_weights, d_dL_doutput, dev_grad_input, batch_size, in_channels,
            out_channels, height, width
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // dL/dweights
    if (d_dL_dweights) {
        int total_weights = out_channels * in_channels * 3 * 3;
        int grid_size_weights = (total_weights + block_size - 1) / block_size;
        conv2d_backward_weights_kernel_opt<<<grid_size_weights, block_size>>>(
            dev_input_data, d_dL_doutput, d_dL_dweights, batch_size, in_channels,
            out_channels, height, width
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // dL/dbias
    if (d_dL_dbias) {
        int grid_size_bias = (out_channels + block_size - 1) / block_size;
        conv2d_backward_bias_kernel_opt<<<grid_size_bias, block_size>>>(
            d_dL_doutput, d_dL_dbias, batch_size, out_channels, height, width
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// OPTIMIZED MAXPOOL KERNELS (2x2 FULLY UNROLLED)
// ============================================================================

__global__ void maxpool2d_forward_kernel_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    const int stride = 2;
    int out_height = in_height / 2;
    int out_width  = in_width / 2;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    if (tid >= total) return;

    int tmp = tid;
    int ow = tmp % out_width;  tmp /= out_width;
    int oh = tmp % out_height; tmp /= out_height;
    int c  = tmp % channels;   tmp /= channels;
    int n  = tmp;

    int base_y = oh * stride;
    int base_x = ow * stride;

    // FULLY UNROLLED 2x2 POOL
    float v0 = -1e38f, v1 = -1e38f, v2 = -1e38f, v3 = -1e38f;

    // dy=0, dx=0
    if (base_y + 0 < in_height && base_x + 0 < in_width) {
        int idx = idx4_opt(n, c, base_y + 0, base_x + 0, channels, in_height, in_width);
        v0 = input[idx];
    }
    // dy=0, dx=1
    if (base_y + 0 < in_height && base_x + 1 < in_width) {
        int idx = idx4_opt(n, c, base_y + 0, base_x + 1, channels, in_height, in_width);
        v1 = input[idx];
    }
    // dy=1, dx=0
    if (base_y + 1 < in_height && base_x + 0 < in_width) {
        int idx = idx4_opt(n, c, base_y + 1, base_x + 0, channels, in_height, in_width);
        v2 = input[idx];
    }
    // dy=1, dx=1
    if (base_y + 1 < in_height && base_x + 1 < in_width) {
        int idx = idx4_opt(n, c, base_y + 1, base_x + 1, channels, in_height, in_width);
        v3 = input[idx];
    }

    float best = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));

    int out_idx = idx4_opt(n, c, oh, ow, channels, out_height, out_width);
    output[out_idx] = best;
}

void gpu_maxpool2d_forward_opt(
    const float* dev_input_data, float* d_output, int batch_size, int channels,
    int in_height, int in_width
) {
    int out_height = in_height / 2;
    int out_width = in_width / 2;
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    maxpool2d_forward_kernel_opt<<<grid_size, block_size>>>(
        dev_input_data, d_output, batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void maxpool2d_backward_kernel_opt(
    const float* input,
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    const int pool_size = 2;
    int out_height = in_height / 2;
    int out_width = in_width / 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_height * out_width;

    if (idx >= total_outputs) return;

    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);

    float out_val = output[idx];
    float grad = dL_doutput[idx];

    // FULLY UNROLLED 2x2 POOL BACKWARD
    // ph=0, pw=0
    {
        int ih = oh * pool_size + 0;
        int iw = ow * pool_size + 0;
        int input_idx = b * (channels * in_height * in_width) +
                        c * (in_height * in_width) + ih * in_width + iw;
        if (input[input_idx] == out_val) {
            atomicAdd(&dL_dinput[input_idx], grad);
        }
    }
    // ph=0, pw=1
    {
        int ih = oh * pool_size + 0;
        int iw = ow * pool_size + 1;
        int input_idx = b * (channels * in_height * in_width) +
                        c * (in_height * in_width) + ih * in_width + iw;
        if (input[input_idx] == out_val) {
            atomicAdd(&dL_dinput[input_idx], grad);
        }
    }
    // ph=1, pw=0
    {
        int ih = oh * pool_size + 1;
        int iw = ow * pool_size + 0;
        int input_idx = b * (channels * in_height * in_width) +
                        c * (in_height * in_width) + ih * in_width + iw;
        if (input[input_idx] == out_val) {
            atomicAdd(&dL_dinput[input_idx], grad);
        }
    }
    // ph=1, pw=1
    {
        int ih = oh * pool_size + 1;
        int iw = ow * pool_size + 1;
        int input_idx = b * (channels * in_height * in_width) +
                        c * (in_height * in_width) + ih * in_width + iw;
        if (input[input_idx] == out_val) {
            atomicAdd(&dL_dinput[input_idx], grad);
        }
    }
}

void gpu_maxpool2d_backward_opt(
    const float* dev_input_data, const float* d_output, const float* d_dL_doutput,
    float* dev_grad_input, int batch_size, int channels, int in_height, int in_width
) {
    int out_height = in_height / 2;
    int out_width = in_width / 2;
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    int input_size = batch_size * channels * in_height * in_width;
    CUDA_CHECK(cudaMemset(dev_grad_input, 0, input_size * sizeof(float)));

    maxpool2d_backward_kernel_opt<<<grid_size, block_size>>>(
        dev_input_data, d_output, d_dL_doutput, dev_grad_input, batch_size, channels,
        in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// OPTIMIZED UPSAMPLE KERNELS (2x2 FULLY UNROLLED)
// ============================================================================

__global__ void upsample2d_backward_kernel_opt(
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    int out_height = in_height * 2;
    int out_width = in_width * 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = batch_size * channels * in_height * in_width;

    if (idx >= total_inputs) return;

    int iw = idx % in_width;
    int ih = (idx / in_width) % in_height;
    int c = (idx / (in_width * in_height)) % channels;
    int b = idx / (in_width * in_height * channels);

    // FULLY UNROLLED 2x2 UPSAMPLE BACKWARD
    float sum = 0.0f;

    // dh=0, dw=0
    {
        int oh = ih * 2 + 0;
        int ow = iw * 2 + 0;
        int output_idx = b * (channels * out_height * out_width) +
                         c * (out_height * out_width) + oh * out_width + ow;
        sum += dL_doutput[output_idx];
    }
    // dh=0, dw=1
    {
        int oh = ih * 2 + 0;
        int ow = iw * 2 + 1;
        int output_idx = b * (channels * out_height * out_width) +
                         c * (out_height * out_width) + oh * out_width + ow;
        sum += dL_doutput[output_idx];
    }
    // dh=1, dw=0
    {
        int oh = ih * 2 + 1;
        int ow = iw * 2 + 0;
        int output_idx = b * (channels * out_height * out_width) +
                         c * (out_height * out_width) + oh * out_width + ow;
        sum += dL_doutput[output_idx];
    }
    // dh=1, dw=1
    {
        int oh = ih * 2 + 1;
        int ow = iw * 2 + 1;
        int output_idx = b * (channels * out_height * out_width) +
                         c * (out_height * out_width) + oh * out_width + ow;
        sum += dL_doutput[output_idx];
    }

    dL_dinput[idx] = sum;
}

void gpu_upsample2d_backward_opt(
    const float* d_dL_doutput, float* dev_grad_input, int batch_size, int channels,
    int in_height, int in_width
) {
    int total_inputs = batch_size * channels * in_height * in_width;
    int block_size = 256;
    int grid_size = (total_inputs + block_size - 1) / block_size;

    upsample2d_backward_kernel_opt<<<grid_size, block_size>>>(
        d_dL_doutput, dev_grad_input, batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// GPUAutoencoderLoopOpt CLASS IMPLEMENTATION
// ============================================================================

GPUAutoencoderLoopOpt::GPUAutoencoderLoopOpt() : GPUAutoencoder() {
    // Inherit everything from parent
}

GPUAutoencoderLoopOpt::~GPUAutoencoderLoopOpt() {
    // Parent destructor handles cleanup
}

void GPUAutoencoderLoopOpt::forward_device(const float* d_in, int batch_size) {
    // Use optimized kernels instead of baseline

    // Encoder
    // Conv1: 3->256, 32x32 + ReLU
    gpu_conv2d_forward_opt(d_in, dev_enc_conv1_w, dev_enc_conv1_b, dev_enc_act1,
                           batch_size, 3, 256, 32, 32);
    gpu_relu_forward(dev_enc_act1, batch_size * 256 * 32 * 32);

    // MaxPool1: 32x32->16x16
    gpu_maxpool2d_forward_opt(dev_enc_act1, dev_enc_pool1, batch_size, 256, 32, 32);

    // Conv2: 256->128, 16x16 + ReLU
    gpu_conv2d_forward_opt(dev_enc_pool1, dev_enc_conv2_w, dev_enc_conv2_b, dev_enc_act2,
                           batch_size, 256, 128, 16, 16);
    gpu_relu_forward(dev_enc_act2, batch_size * 128 * 16 * 16);

    // MaxPool2 (encoded layer): 16x16->8x8
    gpu_maxpool2d_forward_opt(dev_enc_act2, dev_latent, batch_size, 128, 16, 16);

    // Decoder
    // Conv3: 128->128, 8x8 + ReLU
    gpu_conv2d_forward_opt(dev_latent, dev_dec_conv1_w, dev_dec_conv1_b, dev_dec_conv1_out,
                           batch_size, 128, 128, 8, 8);
    gpu_relu_forward(dev_dec_conv1_out, batch_size * 128 * 8 * 8);

    // Upsample1: 8x8->16x16
    gpu_upsample2d_forward(dev_dec_conv1_out, dev_dec_upsample1, batch_size, 128, 8, 8);

    // Conv4: 128->256, 16x16 + ReLU
    gpu_conv2d_forward_opt(dev_dec_upsample1, dev_dec_conv2_w, dev_dec_conv2_b, dev_dec_act1,
                           batch_size, 128, 256, 16, 16);
    gpu_relu_forward(dev_dec_act1, batch_size * 256 * 16 * 16);

    // Upsample2: 16x16->32x32
    gpu_upsample2d_forward(dev_dec_act1, dev_dec_upsample2, batch_size, 256, 16, 16);

    // Conv5: 256->3, 32x32 (output, no activation)
    gpu_conv2d_forward_opt(dev_dec_upsample2, dev_dec_conv3_w, dev_dec_conv3_b, dev_dec_out,
                           batch_size, 256, 3, 32, 32);
}

void GPUAutoencoderLoopOpt::backward_device(const float* d_in, const float* d_tgt, int batch_size) {
    int output_size = batch_size * 3 * 32 * 32;

    // 1. Compute gradient at output
    gpu_mse_loss_gradient(dev_dec_out, d_tgt, dev_grad_dec_out, output_size);

    // 2. Backward through Conv5: 256->3, 32x32
    gpu_conv2d_backward_opt(dev_dec_upsample2, dev_dec_conv3_w, dev_grad_dec_out,
                            dev_grad_dec_upsample2, dev_grad_dec_conv3_w, dev_grad_dec_conv3_b,
                            batch_size, 256, 3, 32, 32);

    // 3. Backward through Upsample2: 16x16->32x32
    gpu_upsample2d_backward_opt(dev_grad_dec_upsample2, dev_grad_dec_act1,
                                batch_size, 256, 16, 16);

    // 4. Backward through ReLU4
    gpu_relu_backward(dev_dec_act1, dev_grad_dec_act1, dev_grad_dec_act1, batch_size * 256 * 16 * 16);

    // 5. Backward through Conv4: 128->256, 16x16
    gpu_conv2d_backward_opt(dev_dec_upsample1, dev_dec_conv2_w, dev_grad_dec_act1,
                            dev_grad_dec_upsample1, dev_grad_dec_conv2_w, dev_grad_dec_conv2_b,
                            batch_size, 128, 256, 16, 16);

    // 6. Backward through Upsample1: 8x8->16x16
    gpu_upsample2d_backward(dev_grad_dec_upsample1, dev_grad_dec_conv1, batch_size, 128, 8, 8);

    // 7. Backward through ReLU3
    gpu_relu_backward(dev_dec_conv1_out, dev_grad_dec_conv1, dev_grad_dec_conv1, batch_size * 128 * 8 * 8);

    // 8. Backward through Conv3: 128->128, 8x8
    gpu_conv2d_backward_opt(dev_latent, dev_dec_conv1_w, dev_grad_dec_conv1,
                            dev_grad_latent, dev_grad_dec_conv1_w, dev_grad_dec_conv1_b,
                            batch_size, 128, 128, 8, 8);

    // 9. Backward through MaxPool2: 16x16->8x8
    gpu_maxpool2d_backward_opt(dev_enc_act2, dev_latent, dev_grad_latent, dev_grad_enc_act2,
                               batch_size, 128, 16, 16);

    // 10. Backward through ReLU2
    gpu_relu_backward(dev_enc_act2, dev_grad_enc_act2, dev_grad_enc_act2, batch_size * 128 * 16 * 16);

    // 11. Backward through Conv2: 256->128, 16x16
    gpu_conv2d_backward_opt(dev_enc_pool1, dev_enc_conv2_w, dev_grad_enc_act2,
                            dev_grad_enc_pool1, dev_grad_enc_conv2_w, dev_grad_enc_conv2_b,
                            batch_size, 256, 128, 16, 16);

    // 12. Backward through MaxPool1: 32x32->16x16
    gpu_maxpool2d_backward_opt(dev_enc_act1, dev_enc_pool1, dev_grad_enc_pool1, dev_grad_enc_act1,
                               batch_size, 256, 32, 32);

    // 13. Backward through ReLU1
    gpu_relu_backward(dev_enc_act1, dev_grad_enc_act1, dev_grad_enc_act1, batch_size * 256 * 32 * 32);

    // 14. Backward through Conv1: 3->256, 32x32
    gpu_conv2d_backward_opt(d_in, dev_enc_conv1_w, dev_grad_enc_act1,
                            dev_grad_input, dev_grad_enc_conv1_w, dev_grad_enc_conv1_b,
                            batch_size, 3, 256, 32, 32);
}

void GPUAutoencoderLoopOpt::extract_features_device(const float* d_in, float* d_features, int batch_size) {
    // Run encoder only with optimized kernels

    // Conv1: 3->256, 32x32 + ReLU
    gpu_conv2d_forward_opt(d_in, dev_enc_conv1_w, dev_enc_conv1_b, dev_enc_act1,
                           batch_size, 3, 256, 32, 32);
    gpu_relu_forward(dev_enc_act1, batch_size * 256 * 32 * 32);

    // MaxPool1: 32x32->16x16
    gpu_maxpool2d_forward_opt(dev_enc_act1, dev_enc_pool1, batch_size, 256, 32, 32);

    // Conv2: 256->128, 16x16 + ReLU
    gpu_conv2d_forward_opt(dev_enc_pool1, dev_enc_conv2_w, dev_enc_conv2_b, dev_enc_act2,
                           batch_size, 256, 128, 16, 16);
    gpu_relu_forward(dev_enc_act2, batch_size * 128 * 16 * 16);

    // MaxPool2 (encoded layer): 16x16->8x8
    gpu_maxpool2d_forward_opt(dev_enc_act2, d_features, batch_size, 128, 16, 16);
}
