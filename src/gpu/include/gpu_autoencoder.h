#pragma once
#include <stdio.h>
#include <cstring>
#include <string>
#include <cuda_runtime.h>
#include <random>


#define IMG_C 3
#define IMG_H 32
#define IMG_W 32


class GPUAutoencoder {
public:
    GPUAutoencoder();
    virtual ~GPUAutoencoder();

    // ======================
    // Public API
    // ======================
    void initialize();
    void forward(const float* h_input, float* h_output, int batch_size);
    void backward(const float* h_input, const float* h_target, int batch_size);
    void update_weights(float learning_rate);
    float compute_loss(const float* h_target, int batch_size);

    void extract_features(const float* h_input, float* h_features, int batch_size);

    void save_weights(const std::string& filepath);
    void load_weights(const std::string& filepath);

protected:

    // ======================
    // Host Weights
    // ======================
    float *host_enc_conv1_w, *host_enc_conv1_b;
    float *host_enc_conv2_w, *host_enc_conv2_b;
    float *host_dec_conv1_w, *host_dec_conv1_b;
    float *host_dec_conv2_w, *host_dec_conv2_b;
    float *host_dec_conv3_w, *host_dec_conv3_b;

    // ======================
    // Device Weights
    // ======================
    float *dev_enc_conv1_w, *dev_enc_conv1_b;
    float *dev_enc_conv2_w, *dev_enc_conv2_b;
    float *dev_dec_conv1_w, *dev_dec_conv1_b;
    float *dev_dec_conv2_w, *dev_dec_conv2_b;
    float *dev_dec_conv3_w, *dev_dec_conv3_b;

    // ======================
    // Device Gradients
    // ======================
    float *dev_grad_enc_conv1_w, *dev_grad_enc_conv1_b;
    float *dev_grad_enc_conv2_w, *dev_grad_enc_conv2_b;
    float *dev_grad_dec_conv1_w, *dev_grad_dec_conv1_b;
    float *dev_grad_dec_conv2_w, *dev_grad_dec_conv2_b;
    float *dev_grad_dec_conv3_w, *dev_grad_dec_conv3_b;

    // ======================
    // Device Activations
    // ======================
    float *dev_input_data, *dev_target_data;
    float *dev_enc_act1, *dev_enc_pool1, *dev_enc_act2, *dev_latent;
    float *dev_dec_conv1_out, *dev_dec_upsample1, *dev_dec_act1, *dev_dec_upsample2, *dev_dec_out;

    // ======================
    // Device Backprop Buffers
    // ======================
    float *dev_grad_dec_out, *dev_grad_dec_outdev_grad_dec_upsample2, *dev_grad_dec_act1, *dev_grad_dec_upsample1;
    float *dev_grad_dec_conv1, *dev_grad_latent, *dev_grad_enc_act2, *dev_grad_enc_pool1;
    float *dev_grad_enc_act1, *dev_grad_input;

    // ======================
    // Batch 
    int batch_size;
    int max_batch_size;
    bool memory_allocated;

    // ======================
    // Internal Helpers
    // ======================
    void allocate_host_memory();
    void free_host_memory();

    void allocate_device_memory(int batch_size);
    void free_device_memory();

    void copy_weights_to_device();
    void copy_weights_to_host();

    virtual void forward_device(const float* d_in, int batch_size);
    virtual void backward_device(const float* d_in, const float* d_tgt, int batch_size);

    virtual void extract_features_device(const float* d_in, float* d_features, int batch_size);
};


// HELPER FUNCTION

static void init_weights_xavier(float* weights, int in_channels, int out_channels);

__global__ void conv2d_forward_kernel(
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

void gpu_conv2d_forward(
    const float* dev_input_data, const float* d_weights, const float* d_bias,
    float* d_output, int batch_size, int in_channels, int out_channels,
    int height, int width
);

__global__ void relu_forward_kernel(float* __restrict__ data, int size);

void gpu_relu_forward(float* d_data, int size);

__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

void gpu_maxpool2d_forward(
    const float* dev_input_data, float* d_output, int batch_size, int channels,
    int in_height, int in_width
);

__global__ void upsample2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

void gpu_upsample2d_forward(
    const float* dev_input_data, float* d_output, int batch_size, int channels,
    int in_height, int in_width
);

__global__ void conv2d_backwardev_input_data_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ dL_doutput,
    float* __restrict__ dL_dinput,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) ;

__global__ void conv2d_backward_weights_kernel(
    const float* __restrict__ input,
    const float* __restrict__ dL_doutput,
    float* __restrict__ dL_dweights,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

__global__ void conv2d_backward_bias_kernel(
    const float* __restrict__ dL_doutput,
    float* __restrict__ dL_dbias,
    int batch_size,
    int out_channels,
    int height,
    int width
);

void gpu_conv2d_backward(
    const float* dev_input_data, const float* d_weights, const float* d_dL_doutput,
    float* dev_grad_input, float* d_dL_dweights, float* d_dL_dbias,
    int batch_size, int in_channels, int out_channels, int height, int width
) ;


__global__ void relu_backward_kernel(
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int size
);

void gpu_relu_backward(
    const float* d_output, const float* d_dL_doutput, float* dev_grad_input, int size
);


__global__ void maxpool2d_backward_kernel(
    const float* input,
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);

void gpu_maxpool2d_backward(
    const float* dev_input_data, const float* d_output, const float* d_dL_doutput,
    float* dev_grad_input, int batch_size, int channels, int in_height, int in_width
);

__global__ void upsample2d_backward_kernel(
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int channels,
    int in_height,
    int in_width
);


void gpu_upsample2d_backward(
    const float* d_dL_doutput, float* dev_grad_input, int batch_size, int channels,
    int in_height, int in_width
);

__global__ void mse_loss_gradient_kernel(
    const float* output,
    const float* target,
    float* dL_doutput,
    int size
);

void gpu_mse_loss_gradient(
    const float* d_output, const float* dev_target_data, float* d_dL_doutput, int size
);

float gpu_mse_loss(const float* d_output, const float* dev_target_data, int size);

__global__ void sgd_update_kernel(
    float* weights,
    const float* dL_dweights,
    float learning_rate,
    float clip_value,
    int size
);

void gpu_sgd_update(
    float* d_weights, float* d_dL_dweights, float learning_rate,
    float clip_value, int size
);




