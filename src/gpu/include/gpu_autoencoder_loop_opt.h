#pragma once
#include "gpu_autoencoder.h"

// ============================================================================
// OPTIMIZED KERNELS WITH LOOP UNROLLING
// ============================================================================

// Optimized Convolution Kernels (3x3 fully unrolled)
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
);

__global__ void conv2d_backward_input_kernel_opt(
    const float* __restrict__ weights,
    const float* __restrict__ dL_doutput,
    float* __restrict__ dL_dinput,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

__global__ void conv2d_backward_weights_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ dL_doutput,
    float* __restrict__ dL_dweights,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

__global__ void conv2d_backward_bias_kernel_opt(
    const float* __restrict__ dL_doutput,
    float* __restrict__ dL_dbias,
    int batch_size,
    int out_channels,
    int height,
    int width
);

// Optimized MaxPool Kernels (2x2 fully unrolled)
__global__ void maxpool2d_forward_kernel_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

__global__ void maxpool2d_backward_kernel_opt(
    const float* input,
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

// Optimized Upsample Kernels (2x2 fully unrolled)
__global__ void upsample2d_backward_kernel_opt(
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

// Wrapper functions for optimized kernels
void gpu_conv2d_forward_opt(
    const float* dev_input_data, const float* d_weights, const float* d_bias,
    float* d_output, int batch_size, int in_channels, int out_channels,
    int height, int width
);

void gpu_conv2d_backward_opt(
    const float* dev_input_data, const float* d_weights, const float* d_dL_doutput,
    float* dev_grad_input, float* d_dL_dweights, float* d_dL_dbias,
    int batch_size, int in_channels, int out_channels, int height, int width
);

void gpu_maxpool2d_forward_opt(
    const float* dev_input_data, float* d_output, int batch_size, int channels,
    int in_height, int in_width
);

void gpu_maxpool2d_backward_opt(
    const float* dev_input_data, const float* d_output, const float* d_dL_doutput,
    float* dev_grad_input, int batch_size, int channels, int in_height, int in_width
);

void gpu_upsample2d_backward_opt(
    const float* d_dL_doutput, float* dev_grad_input, int batch_size, int channels,
    int in_height, int in_width
);

// ============================================================================
// GPUAutoencoderLoopOpt Class
// ============================================================================

class GPUAutoencoderLoopOpt : public GPUAutoencoder {
public:
    GPUAutoencoderLoopOpt();
    virtual ~GPUAutoencoderLoopOpt();

protected:
    // Override device-level functions to use optimized kernels
    void forward_device(const float* d_in, int batch_size) override;
    void backward_device(const float* d_in, const float* d_tgt, int batch_size) override;
    void extract_features_device(const float* d_in, float* d_features, int batch_size) override;
};
