#include <cuda_runtime.h>
#include <stdio.h>

#include <cstring>
#include <random>
#include <string>


// Macro kiểm tra lỗi CUDA đơn giản
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ============================================================================
// Forward Pass Kernels
// ============================================================================

__global__ void conv2d_forward_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    // 3x3 convolution with padding=1, stride=1
    const int kernel_size = 3;
    const int pad = 1;

    // Calculate global thread position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * height * width;

    if (idx >= total_outputs) return;

    // Decompose linear index into (b, oc, h, w)
    int w = idx % width;
    int h = (idx / width) % height;
    int oc = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);

    float sum = bias[oc];

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = h + kh - pad;
                int iw = w + kw - pad;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = b * (in_channels * height * width) +
                                    ic * (height * width) + ih * width + iw;
                    int weight_idx = oc * (in_channels * kernel_size * kernel_size) +
                                     ic * (kernel_size * kernel_size) +
                                     kh * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    output[idx] = sum;
}

void gpu_conv2d_forward(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output, int batch_size, int in_channels, int out_channels,
    int height, int width
) {
    int total_outputs = batch_size * out_channels * height * width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;
    conv2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_weights, d_bias, d_output, batch_size, in_channels,
        out_channels, height, width
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void relu_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (data[idx] > 0.0f) ? data[idx] : 0.0f;
    }
}

void gpu_relu_forward(float* d_data, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    relu_forward_kernel<<<grid_size, block_size>>>(d_data, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void maxpool2d_forward_kernel(
    const float* input,
    float* output,
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

    // Decompose linear index into (b, c, h, w)
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);

    float max_val = -1e38f;

    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = h * pool_size + ph;
            int iw = w * pool_size + pw;
            int input_idx = b * (channels * in_height * in_width) +
                            c * (in_height * in_width) + ih * in_width + iw;
            float val = input[input_idx];
            if (val > max_val) max_val = val;
        }
    }

    output[idx] = max_val;
}

void gpu_maxpool2d_forward(
    const float* d_input, float* d_output, int batch_size, int channels,
    int in_height, int in_width
) {
    int out_height = in_height / 2;
    int out_width = in_width / 2;
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;
    maxpool2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void upsample2d_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    int out_height = in_height * 2;
    int out_width = in_width * 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_height * out_width;

    if (idx >= total_outputs) return;

    // Decompose linear index into (b, c, h, w)
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);

    // Nearest neighbor: map output coord to input coord
    int ih = h / 2;
    int iw = w / 2;

    int input_idx = b * (channels * in_height * in_width) +
                    c * (in_height * in_width) + ih * in_width + iw;

    output[idx] = input[input_idx];
}

void gpu_upsample2d_forward(
    const float* d_input, float* d_output, int batch_size, int channels,
    int in_height, int in_width
) {
    int out_height = in_height * 2;
    int out_width = in_width * 2;
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;
    upsample2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Backward Pass Kernels
// ============================================================================

__global__ void conv2d_backward_input_kernel(
    const float* weights,
    const float* dL_doutput,
    float* dL_dinput,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    const int kernel_size = 3;
    const int pad = 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = batch_size * in_channels * height * width;

    if (idx >= total_inputs) return;

    // Decompose linear index into (b, ic, ih, iw)
    int iw = idx % width;
    int ih = (idx / width) % height;
    int ic = (idx / (width * height)) % in_channels;
    int b = idx / (width * height * in_channels);

    float sum = 0.0f;

    // For each output channel and kernel position that affects this input
    for (int oc = 0; oc < out_channels; oc++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Output position that used this input with this kernel position
                int oh = ih - kh + pad;
                int ow = iw - kw + pad;

                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int output_idx = b * (out_channels * height * width) +
                                     oc * (height * width) + oh * width + ow;
                    int weight_idx = oc * (in_channels * kernel_size * kernel_size) +
                                     ic * (kernel_size * kernel_size) +
                                     kh * kernel_size + kw;
                    sum += dL_doutput[output_idx] * weights[weight_idx];
                }
            }
        }
    }

    dL_dinput[idx] = sum;
}

__global__ void conv2d_backward_weights_kernel(
    const float* input,
    const float* dL_doutput,
    float* dL_dweights,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    const int kernel_size = 3;
    const int pad = 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = out_channels * in_channels * kernel_size * kernel_size;

    if (idx >= total_weights) return;

    // Decompose linear index into (oc, ic, kh, kw)
    int kw = idx % kernel_size;
    int kh = (idx / kernel_size) % kernel_size;
    int ic = (idx / (kernel_size * kernel_size)) % in_channels;
    int oc = idx / (kernel_size * kernel_size * in_channels);

    float sum = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < height; oh++) {
            for (int ow = 0; ow < width; ow++) {
                int ih = oh + kh - pad;
                int iw = ow + kw - pad;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = b * (in_channels * height * width) +
                                    ic * (height * width) + ih * width + iw;
                    int output_idx = b * (out_channels * height * width) +
                                     oc * (height * width) + oh * width + ow;
                    sum += input[input_idx] * dL_doutput[output_idx];
                }
            }
        }
    }

    dL_dweights[idx] = sum;
}

__global__ void conv2d_backward_bias_kernel(
    const float* dL_doutput,
    float* dL_dbias,
    int batch_size,
    int out_channels,
    int height,
    int width
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;

    if (oc >= out_channels) return;

    float sum = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = b * (out_channels * height * width) +
                          oc * (height * width) + h * width + w;
                sum += dL_doutput[idx];
            }
        }
    }

    dL_dbias[oc] = sum;
}

void gpu_conv2d_backward(
    const float* d_input, const float* d_weights, const float* d_dL_doutput,
    float* d_dL_dinput, float* d_dL_dweights, float* d_dL_dbias,
    int batch_size, int in_channels, int out_channels, int height, int width
) {
    int block_size = 256;

    // dL/dinput
    if (d_dL_dinput) {
        int total_inputs = batch_size * in_channels * height * width;
        int grid_size_input = (total_inputs + block_size - 1) / block_size;
        conv2d_backward_input_kernel<<<grid_size_input, block_size>>>(
            d_weights, d_dL_doutput, d_dL_dinput, batch_size, in_channels,
            out_channels, height, width
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // dL/dweights
    if (d_dL_dweights) {
        int total_weights = out_channels * in_channels * 3 * 3;
        int grid_size_weights = (total_weights + block_size - 1) / block_size;
        conv2d_backward_weights_kernel<<<grid_size_weights, block_size>>>(
            d_input, d_dL_doutput, d_dL_dweights, batch_size, in_channels,
            out_channels, height, width
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // dL/dbias
    if (d_dL_dbias) {
        int grid_size_bias = (out_channels + block_size - 1) / block_size;
        conv2d_backward_bias_kernel<<<grid_size_bias, block_size>>>(
            d_dL_doutput, d_dL_dbias, batch_size, out_channels, height, width
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void relu_backward_kernel(
    const float* output,
    const float* dL_doutput,
    float* dL_dinput,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dL_dinput[idx] = (output[idx] > 0.0f) ? dL_doutput[idx] : 0.0f;
    }
}

void gpu_relu_backward(
    const float* d_output, const float* d_dL_doutput, float* d_dL_dinput, int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    relu_backward_kernel<<<grid_size, block_size>>>(
        d_output, d_dL_doutput, d_dL_dinput, size
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void maxpool2d_backward_kernel(
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

    // Decompose linear index into (b, c, oh, ow)
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);

    float out_val = output[idx];
    float grad = dL_doutput[idx];

    // Find which input position had the max value and pass gradient there
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = oh * pool_size + ph;
            int iw = ow * pool_size + pw;
            int input_idx = b * (channels * in_height * in_width) +
                            c * (in_height * in_width) + ih * in_width + iw;
            
            // Pass gradient to the position that had the max value
            if (input[input_idx] == out_val) {
                // Sử dụng atomicAdd vì nhiều khối đầu ra có thể ánh xạ tới cùng một đầu vào
                atomicAdd(&dL_dinput[input_idx], grad);
            }
        }
    }
}

void gpu_maxpool2d_backward(
    const float* d_input, const float* d_output, const float* d_dL_doutput,
    float* d_dL_dinput, int batch_size, int channels, int in_height, int in_width
) {
    int out_height = in_height / 2;
    int out_width = in_width / 2;
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    // Khởi tạo dL_dinput bằng 0 trước khi sử dụng atomicAdd
    int input_size = batch_size * channels * in_height * in_width;
    CUDA_CHECK(cudaMemset(d_dL_dinput, 0, input_size * sizeof(float)));

    maxpool2d_backward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, d_dL_doutput, d_dL_dinput, batch_size, channels,
        in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void upsample2d_backward_kernel(
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

    // Decompose linear index into (b, c, ih, iw)
    int iw = idx % in_width;
    int ih = (idx / in_width) % in_height;
    int c = (idx / (in_width * in_height)) % channels;
    int b = idx / (in_width * in_height * channels);

    // Sum gradients from the 4 output positions that came from this input
    float sum = 0.0f;
    for (int dh = 0; dh < 2; dh++) {
        for (int dw = 0; dw < 2; dw++) {
            int oh = ih * 2 + dh;
            int ow = iw * 2 + dw;
            int output_idx = b * (channels * out_height * out_width) +
                             c * (out_height * out_width) + oh * out_width + ow;
            sum += dL_doutput[output_idx];
        }
    }

    dL_dinput[idx] = sum;
}

void gpu_upsample2d_backward(
    const float* d_dL_doutput, float* d_dL_dinput, int batch_size, int channels,
    int in_height, int in_width
) {
    int total_inputs = batch_size * channels * in_height * in_width;
    int block_size = 256;
    int grid_size = (total_inputs + block_size - 1) / block_size;
    upsample2d_backward_kernel<<<grid_size, block_size>>>(
        d_dL_doutput, d_dL_dinput, batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Loss Kernels
// ============================================================================

__global__ void mse_loss_gradient_kernel(
    const float* output,
    const float* target,
    float* dL_doutput,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dL_doutput[idx] = 2.0f * (output[idx] - target[idx]) / size;
    }
}

void gpu_mse_loss_gradient(
    const float* d_output, const float* d_target, float* d_dL_doutput, int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    mse_loss_gradient_kernel<<<grid_size, block_size>>>(
        d_output, d_target, d_dL_doutput, size
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mse_loss_kernel(
    const float* output,
    const float* target,
    float* partial_sums,
    int size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute squared difference
    float val = 0.0f;
    if (idx < size) {
        float diff = output[idx] - target[idx];
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

float gpu_mse_loss(const float* d_output, const float* d_target, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(float)));

    // 1. Compute partial sums
    mse_loss_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_output, d_target, d_partial_sums, size
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. Copy partial sums to host
    float* h_partial_sums = new float[num_blocks];
    CUDA_CHECK(cudaMemcpy(
        h_partial_sums, d_partial_sums, num_blocks * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // 3. Sum on host
    float total_loss = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_loss += h_partial_sums[i];
    }

    // 4. Clean up and normalize
    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    return total_loss / size;
}

// ============================================================================
// Optimization Kernels
// ============================================================================

__global__ void sgd_update_kernel(
    float* weights,
    const float* dL_dweights,
    float learning_rate,
    float clip_value,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = dL_dweights[idx];
        
        // Gradient Clipping (Value clipping)
        if (grad > clip_value) {
            grad = clip_value;
        } else if (grad < -clip_value) {
            grad = -clip_value;
        }

        // Simple SGD update
        weights[idx] -= learning_rate * grad;
    }
}

void gpu_sgd_update(
    float* d_weights, float* d_dL_dweights, float learning_rate,
    float clip_value, int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    sgd_update_kernel<<<grid_size, block_size>>>(
        d_weights, d_dL_dweights, learning_rate, clip_value, size
    );
    CUDA_CHECK(cudaGetLastError());
}


// Weight sizes
constexpr int W1_SIZE = 256 * 3 * 3 * 3; 
constexpr int B1_SIZE = 256;
constexpr int W2_SIZE = 128 * 256 * 3 * 3;
constexpr int B2_SIZE = 128;
constexpr int W3_SIZE = 128 * 128 * 3 * 3;
constexpr int B3_SIZE = 128;
constexpr int W4_SIZE = 256 * 128 * 3 * 3;
constexpr int B4_SIZE = 256;
constexpr int W5_SIZE = 3 * 256 * 3 * 3; 
constexpr int B5_SIZE = 3;

// Xavier weight initialization
static void init_weights_xavier(float* weights, int in_channels, int out_channels) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // Lấy kích thước kernel 3x3
    float limit = std::sqrt(6.0f / ((in_channels * 3 * 3) + (out_channels * 3 * 3)));
    std::uniform_real_distribution<float> dis(-limit, limit);

    int kernel_size = 3 * 3;
    int total_weights = out_channels * in_channels * kernel_size;

    for (int i = 0; i < total_weights; i++) {
        weights[i] = dis(gen);
    }
}

// Khai báo lại class (để biên dịch độc lập, mặc dù ban đầu nó nằm trong header)
class GPUAutoencoder {
public:
    GPUAutoencoder();
    ~GPUAutoencoder();

    void initialize();
    void forward(const float* h_input, float* h_output, int batch_size);
    void backward(const float* h_input, const float* h_target, int batch_size);
    void update_weights(float learning_rate);
    float compute_loss(const float* h_target, int batch_size);
    void extract_features(const float* h_input, float* h_features, int batch_size);
    void save_weights(const std::string& filepath);
    void load_weights(const std::string& filepath);

private:
    // Host pointers
    float *h_w1, *h_b1, *h_w2, *h_b2, *h_w3, *h_b3, *h_w4, *h_b4, *h_w5, *h_b5;

    // Device weight pointers
    float *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_w4, *d_b4, *d_w5, *d_b5;

    // Device gradient pointers
    float *d_dw1, *d_db1, *d_dw2, *d_db2, *d_dw3, *d_db3, *d_dw4, *d_db4, *d_dw5, *d_db5;

    // Device activation pointers
    float *d_input, *d_target;
    float *d_act1, *d_pool1, *d_act2, *d_act3; // act3 là latent
    float *d_conv3_out, *d_up1, *d_act4, *d_up2, *d_act5; // act5 là output

    // Device gradient buffers
    float *d_dL_dact5, *d_dL_dup2, *d_dL_dact4, *d_dL_dup1;
    float *d_dL_dconv3, *d_dL_dact3, *d_dL_dact2, *d_dL_dpool1;
    float *d_dL_dact1, *d_dL_dinput;

    int current_batch_size;
    int max_batch_size;
    bool memory_allocated;

    void allocate_host_memory();
    void free_host_memory();
    void allocate_device_memory(int batch_size);
    void free_device_memory();
    void copy_weights_to_device();
    void copy_weights_to_host();
    void forward_device(const float* d_in, int batch_size);
    void backward_device(const float* d_in, const float* d_tgt, int batch_size);
    void extract_features_device(const float* d_in, float* d_features, int batch_size);
};

// Triển khai class...

GPUAutoencoder::GPUAutoencoder() {
    // Host pointers
    h_w1 = h_b1 = h_w2 = h_b2 = h_w3 = h_b3 = h_w4 = h_b4 = h_w5 = h_b5 = nullptr;

    // Device weight pointers
    d_w1 = d_b1 = d_w2 = d_b2 = d_w3 = d_b3 = d_w4 = d_b4 = d_w5 = d_b5 = nullptr;

    // Device gradient pointers
    d_dw1 = d_db1 = d_dw2 = d_db2 = d_dw3 = d_db3 = d_dw4 = d_db4 = d_dw5 = d_db5 = nullptr;

    // Device activation pointers
    d_input = d_target = nullptr;
    d_act1 = d_pool1 = d_act2 = d_act3 = nullptr;
    d_conv3_out = d_up1 = d_act4 = d_up2 = d_act5 = nullptr;

    // Device gradient buffers
    d_dL_dact5 = d_dL_dup2 = d_dL_dact4 = d_dL_dup1 = nullptr;
    d_dL_dconv3 = d_dL_dact3 = d_dL_dact2 = d_dL_dpool1 = nullptr;
    d_dL_dact1 = d_dL_dinput = nullptr;

    current_batch_size = 0;
    max_batch_size = 64;  // Default max batch size
    memory_allocated = false;
}

GPUAutoencoder::~GPUAutoencoder() {
    free_device_memory();
    free_host_memory();
}

void GPUAutoencoder::allocate_host_memory() {
    if (h_w1) return; // Đã cấp phát

    h_w1 = new float[W1_SIZE];
    h_b1 = new float[B1_SIZE];
    h_w2 = new float[W2_SIZE];
    h_b2 = new float[B2_SIZE];
    h_w3 = new float[W3_SIZE];
    h_b3 = new float[B3_SIZE];
    h_w4 = new float[W4_SIZE];
    h_b4 = new float[B4_SIZE];
    h_w5 = new float[W5_SIZE];
    h_b5 = new float[B5_SIZE];
}

void GPUAutoencoder::free_host_memory() {
    delete[] h_w1; h_w1 = nullptr;
    delete[] h_b1; h_b1 = nullptr;
    delete[] h_w2; h_w2 = nullptr;
    delete[] h_b2; h_b2 = nullptr;
    delete[] h_w3; h_w3 = nullptr;
    delete[] h_b3; h_b3 = nullptr;
    delete[] h_w4; h_w4 = nullptr;
    delete[] h_b4; h_b4 = nullptr;
    delete[] h_w5; h_w5 = nullptr;
    delete[] h_b5; h_b5 = nullptr;
}

void GPUAutoencoder::allocate_device_memory(int batch_size) {
    if (memory_allocated && batch_size <= max_batch_size) {
        current_batch_size = batch_size;
        return;
    }

    if (memory_allocated) {
        free_device_memory();
    }

    max_batch_size = batch_size;
    current_batch_size = batch_size;

    // Allocate device weights
    CUDA_CHECK(cudaMalloc(&d_w1, W1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, B1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, W2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, B2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w3, W3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, B3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w4, W4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b4, B4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w5, W5_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b5, B5_SIZE * sizeof(float)));

    // Allocate device gradients
    CUDA_CHECK(cudaMalloc(&d_dw1, W1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, B1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw2, W2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, B2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw3, W3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db3, B3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw4, W4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db4, B4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw5, W5_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db5, B5_SIZE * sizeof(float)));

    // Allocate device activations
    // Input: batch x 3 x 32 x 32
    CUDA_CHECK(cudaMalloc(&d_input, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    
    // act1: batch x 256 x 32 x 32
    CUDA_CHECK(cudaMalloc(&d_act1, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    
    // pool1: batch x 256 x 16 x 16
    CUDA_CHECK(cudaMalloc(&d_pool1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    
    // act2: batch x 128 x 16 x 16
    CUDA_CHECK(cudaMalloc(&d_act2, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    
    // act3 (latent): batch x 128 x 8 x 8
    CUDA_CHECK(cudaMalloc(&d_act3, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    
    // conv3_out: batch x 128 x 8 x 8
    CUDA_CHECK(cudaMalloc(&d_conv3_out, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    
    // up1: batch x 128 x 16 x 16
    CUDA_CHECK(cudaMalloc(&d_up1, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    
    // act4: batch x 256 x 16 x 16
    CUDA_CHECK(cudaMalloc(&d_act4, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    
    // up2: batch x 256 x 32 x 32
    CUDA_CHECK(cudaMalloc(&d_up2, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    
    // act5 (output): batch x 3 x 32 x 32
    CUDA_CHECK(cudaMalloc(&d_act5, max_batch_size * 3 * 32 * 32 * sizeof(float)));

    // Allocate gradient buffers for backward pass
    CUDA_CHECK(cudaMalloc(&d_dL_dact5, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dup2, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact4, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dup1, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dconv3, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact3, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact2, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dpool1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dact1, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dinput, max_batch_size * 3 * 32 * 32 * sizeof(float)));

    memory_allocated = true;
}

void GPUAutoencoder::free_device_memory() {
    if (!memory_allocated) return;

    // Free device weights
    if (d_w1) cudaFree(d_w1); d_w1 = nullptr;
    if (d_b1) cudaFree(d_b1); d_b1 = nullptr;
    if (d_w2) cudaFree(d_w2); d_w2 = nullptr;
    if (d_b2) cudaFree(d_b2); d_b2 = nullptr;
    if (d_w3) cudaFree(d_w3); d_w3 = nullptr;
    if (d_b3) cudaFree(d_b3); d_b3 = nullptr;
    if (d_w4) cudaFree(d_w4); d_w4 = nullptr;
    if (d_b4) cudaFree(d_b4); d_b4 = nullptr;
    if (d_w5) cudaFree(d_w5); d_w5 = nullptr;
    if (d_b5) cudaFree(d_b5); d_b5 = nullptr;

    // Free device gradients
    if (d_dw1) cudaFree(d_dw1); d_dw1 = nullptr;
    if (d_db1) cudaFree(d_db1); d_db1 = nullptr;
    if (d_dw2) cudaFree(d_dw2); d_dw2 = nullptr;
    if (d_db2) cudaFree(d_db2); d_db2 = nullptr;
    if (d_dw3) cudaFree(d_dw3); d_dw3 = nullptr;
    if (d_db3) cudaFree(d_db3); d_db3 = nullptr;
    if (d_dw4) cudaFree(d_dw4); d_dw4 = nullptr;
    if (d_db4) cudaFree(d_db4); d_db4 = nullptr;
    if (d_dw5) cudaFree(d_dw5); d_dw5 = nullptr;
    if (d_db5) cudaFree(d_db5); d_db5 = nullptr;

    // Free device activations
    if (d_input) cudaFree(d_input); d_input = nullptr;
    if (d_target) cudaFree(d_target); d_target = nullptr;
    if (d_act1) cudaFree(d_act1); d_act1 = nullptr;
    if (d_pool1) cudaFree(d_pool1); d_pool1 = nullptr;
    if (d_act2) cudaFree(d_act2); d_act2 = nullptr;
    if (d_act3) cudaFree(d_act3); d_act3 = nullptr;
    if (d_conv3_out) cudaFree(d_conv3_out); d_conv3_out = nullptr;
    if (d_up1) cudaFree(d_up1); d_up1 = nullptr;
    if (d_act4) cudaFree(d_act4); d_act4 = nullptr;
    if (d_up2) cudaFree(d_up2); d_up2 = nullptr;
    if (d_act5) cudaFree(d_act5); d_act5 = nullptr;

    // Free gradient buffers
    if (d_dL_dact5) cudaFree(d_dL_dact5); d_dL_dact5 = nullptr;
    if (d_dL_dup2) cudaFree(d_dL_dup2); d_dL_dup2 = nullptr;
    if (d_dL_dact4) cudaFree(d_dL_dact4); d_dL_dact4 = nullptr;
    if (d_dL_dup1) cudaFree(d_dL_dup1); d_dL_dup1 = nullptr;
    if (d_dL_dconv3) cudaFree(d_dL_dconv3); d_dL_dconv3 = nullptr;
    if (d_dL_dact3) cudaFree(d_dL_dact3); d_dL_dact3 = nullptr;
    if (d_dL_dact2) cudaFree(d_dL_dact2); d_dL_dact2 = nullptr;
    if (d_dL_dpool1) cudaFree(d_dL_dpool1); d_dL_dpool1 = nullptr;
    if (d_dL_dact1) cudaFree(d_dL_dact1); d_dL_dact1 = nullptr;
    if (d_dL_dinput) cudaFree(d_dL_dinput); d_dL_dinput = nullptr;

    memory_allocated = false;
}

void GPUAutoencoder::copy_weights_to_device() {
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1, W1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1, B1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2, W2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2, B2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w3, h_w3, W3_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3, B3_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w4, h_w4, W4_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b4, h_b4, B4_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w5, h_w5, W5_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b5, h_b5, B5_SIZE * sizeof(float), cudaMemcpyHostToDevice));
}

void GPUAutoencoder::copy_weights_to_host() {
    CUDA_CHECK(cudaMemcpy(h_w1, d_w1, W1_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b1, d_b1, B1_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w2, d_w2, W2_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b2, d_b2, B2_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w3, d_w3, W3_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b3, d_b3, B3_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w4, d_w4, W4_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b4, d_b4, B4_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w5, d_w5, W5_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b5, d_b5, B5_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPUAutoencoder::initialize() {
    allocate_host_memory();

    // Initialize weights using Xavier initialization
    // Chú ý: Kích thước đầu vào/đầu ra cho Xavier là số kênh
    init_weights_xavier(h_w1, 3, 256);
    init_weights_xavier(h_w2, 256, 128);
    init_weights_xavier(h_w3, 128, 128);
    init_weights_xavier(h_w4, 128, 256);
    init_weights_xavier(h_w5, 256, 3);

    // Initialize biases to zero
    memset(h_b1, 0, B1_SIZE * sizeof(float));
    memset(h_b2, 0, B2_SIZE * sizeof(float));
    memset(h_b3, 0, B3_SIZE * sizeof(float));
    memset(h_b4, 0, B4_SIZE * sizeof(float));
    memset(h_b5, 0, B5_SIZE * sizeof(float));

    // Allocate device memory and copy weights
    allocate_device_memory(max_batch_size);
    copy_weights_to_device();
}

void GPUAutoencoder::forward_device(const float* d_in, int batch_size) {
    current_batch_size = batch_size;

    // Encoder
    // Conv1: 3->256, 32x32 + ReLU
    gpu_conv2d_forward(d_in, d_w1, d_b1, d_act1, batch_size, 3, 256, 32, 32);
    gpu_relu_forward(d_act1, batch_size * 256 * 32 * 32);

    // MaxPool1: 32x32->16x16
    gpu_maxpool2d_forward(d_act1, d_pool1, batch_size, 256, 32, 32);

    // Conv2: 256->128, 16x16 + ReLU
    gpu_conv2d_forward(d_pool1, d_w2, d_b2, d_act2, batch_size, 256, 128, 16, 16);
    gpu_relu_forward(d_act2, batch_size * 128 * 16 * 16);

    // MaxPool2 (encoded layer): 16x16->8x8
    gpu_maxpool2d_forward(d_act2, d_act3, batch_size, 128, 16, 16);

    // Decoder
    // Conv3: 128->128, 8x8 + ReLU
    gpu_conv2d_forward(d_act3, d_w3, d_b3, d_conv3_out, batch_size, 128, 128, 8, 8);
    gpu_relu_forward(d_conv3_out, batch_size * 128 * 8 * 8);

    // Upsample1: 8x8->16x16
    gpu_upsample2d_forward(d_conv3_out, d_up1, batch_size, 128, 8, 8);

    // Conv4: 128->256, 16x16 + ReLU
    gpu_conv2d_forward(d_up1, d_w4, d_b4, d_act4, batch_size, 128, 256, 16, 16);
    gpu_relu_forward(d_act4, batch_size * 256 * 16 * 16);

    // Upsample2: 16x16->32x32
    gpu_upsample2d_forward(d_act4, d_up2, batch_size, 256, 16, 16);

    // Conv5: 256->3, 32x32 (output, no activation)
    gpu_conv2d_forward(d_up2, d_w5, d_b5, d_act5, batch_size, 256, 3, 32, 32);
}

void GPUAutoencoder::forward(const float* h_input, float* h_output, int batch_size) {
    // Ensure device memory is allocated
    allocate_device_memory(batch_size);
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));

    // Run forward pass on device
    forward_device(d_input, batch_size);

    // Copy output back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_act5, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPUAutoencoder::backward_device(const float* d_in, const float* d_tgt, int batch_size) {
    int output_size = batch_size * 3 * 32 * 32;

    // 1. Compute gradient at output: dL/dact5 = 2(act5 - target) / N
    gpu_mse_loss_gradient(d_act5, d_tgt, d_dL_dact5, output_size);

    // 2. Backward through Conv5: 256->3, 32x32
    // d_dL_dinput cho Conv5 là d_dL_dup2
    gpu_conv2d_backward(d_up2, d_w5, d_dL_dact5, d_dL_dup2, d_dw5, d_db5,
                        batch_size, 256, 3, 32, 32);

    // 3. Backward through Upsample2: 16x16->32x32
    // d_dL_dinput cho Upsample2 là d_dL_dact4
    gpu_upsample2d_backward(d_dL_dup2, d_dL_dact4, batch_size, 256, 16, 16);

    // 4. Backward through ReLU4
    gpu_relu_backward(d_act4, d_dL_dact4, d_dL_dact4, batch_size * 256 * 16 * 16);

    // 5. Backward through Conv4: 128->256, 16x16
    // d_dL_dinput cho Conv4 là d_dL_dup1
    gpu_conv2d_backward(d_up1, d_w4, d_dL_dact4, d_dL_dup1, d_dw4, d_db4,
                        batch_size, 128, 256, 16, 16);

    // 6. Backward through Upsample1: 8x8->16x16
    // d_dL_dinput cho Upsample1 là d_dL_dconv3
    gpu_upsample2d_backward(d_dL_dup1, d_dL_dconv3, batch_size, 128, 8, 8);

    // 7. Backward through ReLU3
    gpu_relu_backward(d_conv3_out, d_dL_dconv3, d_dL_dconv3, batch_size * 128 * 8 * 8);

    // 8. Backward through Conv3: 128->128, 8x8
    // d_dL_dinput cho Conv3 là d_dL_dact3
    gpu_conv2d_backward(d_act3, d_w3, d_dL_dconv3, d_dL_dact3, d_dw3, d_db3,
                        batch_size, 128, 128, 8, 8);

    // 9. Backward through MaxPool2: 16x16->8x8
    // d_dL_dinput cho MaxPool2 là d_dL_dact2
    gpu_maxpool2d_backward(d_act2, d_act3, d_dL_dact3, d_dL_dact2,
                           batch_size, 128, 16, 16);

    // 10. Backward through ReLU2
    gpu_relu_backward(d_act2, d_dL_dact2, d_dL_dact2, batch_size * 128 * 16 * 16);

    // 11. Backward through Conv2: 256->128, 16x16
    // d_dL_dinput cho Conv2 là d_dL_dpool1
    gpu_conv2d_backward(d_pool1, d_w2, d_dL_dact2, d_dL_dpool1, d_dw2, d_db2,
                        batch_size, 256, 128, 16, 16);

    // 12. Backward through MaxPool1: 32x32->16x16
    // d_dL_dinput cho MaxPool1 là d_dL_dact1
    gpu_maxpool2d_backward(d_act1, d_pool1, d_dL_dpool1, d_dL_dact1,
                           batch_size, 256, 32, 32);

    // 13. Backward through ReLU1
    gpu_relu_backward(d_act1, d_dL_dact1, d_dL_dact1, batch_size * 256 * 32 * 32);

    // 14. Backward through Conv1: 3->256, 32x32
    // d_dL_dinput cho Conv1 là d_dL_dinput
    gpu_conv2d_backward(d_in, d_w1, d_dL_dact1, d_dL_dinput, d_dw1, d_db1,
                        batch_size, 3, 256, 32, 32);
}

void GPUAutoencoder::backward(const float* h_input, const float* h_target, int batch_size) {
    // Ensure device memory is allocated
    allocate_device_memory(batch_size);
    
    // Copy input and target to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, h_target, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));

    // Run forward pass
    forward_device(d_input, batch_size);

    // Run backward pass on device
    backward_device(d_input, d_target, batch_size);
}

void GPUAutoencoder::update_weights(float learning_rate) {
    const float clip_value = 1.0f;

    // Update all weights and biases using SGD with gradient clipping
    gpu_sgd_update(d_w1, d_dw1, learning_rate, clip_value, W1_SIZE);
    gpu_sgd_update(d_b1, d_db1, learning_rate, clip_value, B1_SIZE);
    gpu_sgd_update(d_w2, d_dw2, learning_rate, clip_value, W2_SIZE);
    gpu_sgd_update(d_b2, d_db2, learning_rate, clip_value, B2_SIZE);
    gpu_sgd_update(d_w3, d_dw3, learning_rate, clip_value, W3_SIZE);
    gpu_sgd_update(d_b3, d_db3, learning_rate, clip_value, B3_SIZE);
    gpu_sgd_update(d_w4, d_dw4, learning_rate, clip_value, W4_SIZE);
    gpu_sgd_update(d_b4, d_db4, learning_rate, clip_value, B4_SIZE);
    gpu_sgd_update(d_w5, d_dw5, learning_rate, clip_value, W5_SIZE);
    gpu_sgd_update(d_b5, d_db5, learning_rate, clip_value, B5_SIZE);
}

float GPUAutoencoder::compute_loss(const float* h_target, int batch_size) {
    // Copy target to device
    CUDA_CHECK(cudaMemcpy(d_target, h_target, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
    
    int size = batch_size * 3 * 32 * 32;
    return gpu_mse_loss(d_act5, d_target, size);
}

void GPUAutoencoder::extract_features_device(const float* d_in, float* d_features, int batch_size) {
    // Run encoder only
    // Conv1: 3->256, 32x32 + ReLU
    gpu_conv2d_forward(d_in, d_w1, d_b1, d_act1, batch_size, 3, 256, 32, 32);
    gpu_relu_forward(d_act1, batch_size * 256 * 32 * 32);

    // MaxPool1: 32x32->16x16
    gpu_maxpool2d_forward(d_act1, d_pool1, batch_size, 256, 32, 32);

    // Conv2: 256->128, 16x16 + ReLU
    gpu_conv2d_forward(d_pool1, d_w2, d_b2, d_act2, batch_size, 256, 128, 16, 16);
    gpu_relu_forward(d_act2, batch_size * 128 * 16 * 16);

    // MaxPool2 (encoded layer): 16x16->8x8
    // Gán d_features = d_act3 (output của MaxPool2)
    gpu_maxpool2d_forward(d_act2, d_features, batch_size, 128, 16, 16);
}

void GPUAutoencoder::extract_features(const float* h_input, float* h_features, int batch_size) {
    // Ensure device memory is allocated
    allocate_device_memory(batch_size);
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));

    // Run encoder on device
    extract_features_device(d_input, d_act3, batch_size);

    // Copy features back to host (128 x 8 x 8 = 8192 per image)
    CUDA_CHECK(cudaMemcpy(h_features, d_act3, batch_size * 128 * 8 * 8 * sizeof(float), cudaMemcpyDeviceToHost));
}

// void GPUAutoencoder::save_weights(const std::string& filepath) {
//     // Copy weights from device to host
//     copy_weights_to_host();

//     FILE* f = fopen(filepath.c_str(), "wb");
//     if (!f) {
//         fprintf(stderr, "Failed to open file for writing: %s\n", filepath.c_str());
//         return;
//     }

//     fwrite(h_w1, sizeof(float), W1_SIZE, f);
//     fwrite(h_b1, sizeof(float), B1_SIZE, f);
//     fwrite(h_w2, sizeof(float), W2_SIZE, f);
//     fwrite(h_b2, sizeof(float), B2_SIZE, f);
//     fwrite(h_w3, sizeof(float), W3_SIZE, f);
//     fwrite(h_b3, sizeof(float), B3_SIZE, f);
//     fwrite(h_w4, sizeof(float), W4_SIZE, f);
//     fwrite(h_b4, sizeof(float), B4_SIZE, f);
//     fwrite(h_w5, sizeof(float), W5_SIZE, f);
//     fwrite(h_b5, sizeof(float), B5_SIZE, f);

//     fclose(f);
//     printf("GPU Model weights saved to: %s\n", filepath.c_str());
// }

void GPUAutoencoder::load_weights(const std::string& filepath) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file for reading: %s\n", filepath.c_str());
        return;
    }

    if (!h_w1) {
        allocate_host_memory();
    }

    fread(h_w1, sizeof(float), W1_SIZE, f);
    fread(h_b1, sizeof(float), B1_SIZE, f);
    fread(h_w2, sizeof(float), W2_SIZE, f);
    fread(h_b2, sizeof(float), B2_SIZE, f);
    fread(h_w3, sizeof(float), W3_SIZE, f);
    fread(h_b3, sizeof(float), B3_SIZE, f);
    fread(h_w4, sizeof(float), W4_SIZE, f);
    fread(h_b4, sizeof(float), B4_SIZE, f);
    fread(h_w5, sizeof(float), W5_SIZE, f);
    fread(h_b5, sizeof(float), B5_SIZE, f);

    fclose(f);

    // Allocate device memory if needed and copy weights
    if (!memory_allocated) {
        allocate_device_memory(max_batch_size);
    }
    copy_weights_to_device();

    printf("GPU Model weights loaded from: %s\n", filepath.c_str());
}


// Khai báo các hằng số và hàm tiện ích giả định (Tùy thuộc vào project của bạn)
// Các hằng số này thường được định nghĩa trong một tệp header chung (ví dụ: config.h)
#define IMG_C 3  // Số kênh màu (Color Channels)
#define IMG_H 32 // Chiều cao (Height)
#define IMG_W 32 // Chiều rộng (Width)

// Khai báo các hàm tải/lưu dữ liệu giả định
bool load_cifar10_images(
    const std::string& filename,
    std::vector<std::vector<float>>& images,
    std::vector<int>& labels,
    int num_to_load
);

void save_images_to_binary(
    const std::vector<std::vector<float>>& images,
    const std::string& filename
);


// --- HÀM MAIN ---
int main() {

    // ================================
    // 1. Tham số huấn luyện
    // ================================
    int batch_size = 16;
    int epochs = 5;
    float lr = 1e-5;

    GPUAutoencoder gpu_model;
    gpu_model.initialize();

    // ================================
    // 2. Tải dữ liệu CIFAR
    // ================================
    std::vector<std::vector<float>> train_images;
    std::vector<int> train_labels;

    printf("Loading training data...\n");
    if (!load_cifar10_images(
            "../../data/cifar-100-binary/cifar-100-binary/train.bin",
            train_images,
            train_labels,
            50000))
    {
        printf("Load data failed! Check file path.\n");
        return 1;
    }

    printf("Loaded %zu training images.\n", train_images.size());

    // ================================
    // 3. Bộ nhớ HOST cho batch
    // ================================
    size_t one_img = (size_t)IMG_C * IMG_H * IMG_W;         
    size_t input_sz = (size_t)batch_size * one_img;        

    float *h_input  = (float*)malloc(input_sz * sizeof(float));
    float *h_output = (float*)malloc(input_sz * sizeof(float));

    if (!h_input || !h_output) {
        printf("Host malloc failed!\n");
        return 1;
    }

    // ================================
    // 4. Train Loop
    // ================================
    for (int e = 0; e < epochs; ++e) {
        printf("\n=== Epoch %d ===\n", e + 1);

        float epoch_loss = 0.0f;
        size_t num_batches = 0;

        for (size_t i = 0; i + batch_size <= train_images.size(); i += batch_size)
        {
            // ----- 4.1. Copy batch → h_input -----
            for (int b = 0; b < batch_size; b++) {
                memcpy(
                    &h_input[b * one_img],
                    train_images[i + b].data(),
                    one_img * sizeof(float)
                );
            }

            // ----- 4.2. Forward -----
            gpu_model.forward(h_input, h_output, batch_size);

            // ----- 4.3. Compute Loss -----
            float h_loss = gpu_model.compute_loss(h_input, batch_size);
            gpu_model.backward(h_input, h_input, batch_size);
            gpu_model.update_weights(lr);
            epoch_loss += h_loss;
            num_batches++;
        }

        if (num_batches > 0)
            printf("Average Epoch %d loss = %f\n", e + 1, epoch_loss / num_batches);
        else
            printf("Epoch %d: No batches processed.\n", e + 1);
    }

    // ================================
    // 5. Giải phóng
    // ================================
    free(h_input);
    free(h_output);

    printf("\nDone training.\n");
    return 0;
}
