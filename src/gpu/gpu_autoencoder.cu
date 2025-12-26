#include"gpu_autoencoder.h"
#include <algorithm>

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
// Tiling parameters for shared-memory convolution
#define TILE_W 16
#define TILE_H 16
// Index helper for NCHW layout
__host__ __device__ inline int idx4(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

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
) {
    // 3x3 convolution with padding=1, stride=1
    const int kernel_size = 3;
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

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int iy = oh * stride + ky - pad;
                int ix = ow * stride + kx - pad;
                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    int in_idx = idx4(n, ic, iy, ix, in_channels, height, width);
                    int w_idx  = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                    acc += input[in_idx] * weights[w_idx];
                }
            }
        }
    }

    int out_idx = idx4(n, oc, oh, ow, out_channels, height, width);
    output[out_idx] = acc + bias[oc];  // add bias ngay tại đây
}

// Shared-memory tiled convolution (3x3 kernel, padding=1, stride=1)
__global__ void conv2d_forward_tiled_kernel(
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
    constexpr int K = 3;
    constexpr int P = 1; // padding

    // block tile origin in output coordinates
    int tile_x = blockIdx.x * TILE_W;
    int tile_y = blockIdx.y * TILE_H;

    // blockIdx.z encodes (n * out_channels + oc)
    int n_oc = blockIdx.z;
    int n = n_oc / out_channels;
    int oc = n_oc % out_channels;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Shared memory tile (for one input channel at a time)
    __shared__ float sh[(TILE_H + 2) * (TILE_W + 2)];

    // Each thread will compute one output position within the tile (if in range)
    int out_x = tile_x + tx;
    int out_y = tile_y + ty;

    float acc = 0.0f;

    // Loop over input channels, accumulate contribution
    for (int ic = 0; ic < in_channels; ++ic) {
        // Cooperative load into shared memory: we need to load a (TILE_H+2)x(TILE_W+2)
        // patch of input for this ic and batch n, accounting for padding.
        for (int y = ty; y < TILE_H + 2; y += blockDim.y) {
            for (int x = tx; x < TILE_W + 2; x += blockDim.x) {
                int in_x = tile_x + x - P;
                int in_y = tile_y + y - P;
                float v = 0.0f;
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    int in_idx = idx4(n, ic, in_y, in_x, in_channels, height, width);
                    v = input[in_idx];
                }
                sh[y * (TILE_W + 2) + x] = v;
            }
        }

        __syncthreads();

        // Now compute convolution for the thread's output position (if inside image)
        if (out_x < width && out_y < height) {
            // For 3x3 kernel
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    int sx = tx + kx;
                    int sy = ty + ky;
                    float in_val = sh[sy * (TILE_W + 2) + sx];
                    int w_idx = ((oc * in_channels + ic) * K + ky) * K + kx;
                    acc += in_val * weights[w_idx];
                }
            }
        }

        __syncthreads();
    }

    // Add bias and write output
    if (out_x < width && out_y < height) {
        int out_idx = idx4(n, oc, out_y, out_x, out_channels, height, width);
        output[out_idx] = acc + bias[oc];
    }
}

void gpu_conv2d_forward(
    const float* dev_input_data, const float* d_weights, const float* d_bias,
    float* d_output, int batch_size, int in_channels, int out_channels,
    int height, int width
) {
    // Launch tiled shared-memory kernel: grid.z encodes batch * out_channels
    dim3 block(TILE_W, TILE_H);
    dim3 grid((width + TILE_W - 1) / TILE_W,
              (height + TILE_H - 1) / TILE_H,
              batch_size * out_channels);

    conv2d_forward_tiled_kernel<<<grid, block>>>(
        dev_input_data, d_weights, d_bias, d_output,
        batch_size, in_channels, out_channels, height, width
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void relu_forward_kernel(float* __restrict__ data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float v = data[tid];
        data[tid] = v > 0.0f ? v : 0.0f;
    }
}

void gpu_relu_forward(float* d_data, int size) {
    int block_size = 512;
    int grid_size = (size + block_size - 1) / block_size;
    relu_forward_kernel<<<grid_size, block_size>>>(d_data, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    const int pool_size = 2;
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

    float best = -1e38f;
    int base_y = oh * stride;
    int base_x = ow * stride;

    for (int dy = 0; dy < pool_size; ++dy) {
        for (int dx = 0; dx < pool_size; ++dx) {
            int y = base_y + dy;
            int x = base_x + dx;
            if (y < in_height && x < in_width) {
                int idx = idx4(n, c, y, x, channels, in_height, in_width);
                float v = input[idx];
                if (v > best) best = v;
            }
        }
    }

    int out_idx = idx4(n, c, oh, ow, channels, out_height, out_width);
    output[out_idx] = best;
}

void gpu_maxpool2d_forward(
    const float* dev_input_data, float* d_output, int batch_size, int channels,
    int in_height, int in_width
) {
    int out_height = in_height / 2;
    int out_width = in_width / 2;
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 512;
    int grid_size = (total_outputs + block_size - 1) / block_size;
    maxpool2d_forward_kernel<<<grid_size, block_size>>>(
        dev_input_data, d_output, batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void upsample2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width
) {
    int out_height = in_height * 2;
    int out_width  = in_width * 2;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    if (tid >= total) return;

    int tmp = tid;
    int ow = tmp % out_width;  tmp /= out_width;
    int oh = tmp % out_height; tmp /= out_height;
    int c  = tmp % channels;   tmp /= channels;
    int n  = tmp;

    int ih = oh / 2;
    int iw = ow / 2;

    int in_idx = idx4(n, c, ih, iw, channels, in_height, in_width);
    int out_idx = idx4(n, c, oh, ow, channels, out_height, out_width);
    output[out_idx] = input[in_idx];
}

void gpu_upsample2d_forward(
    const float* dev_input_data, float* d_output, int batch_size, int channels,
    int in_height, int in_width
) {
    int out_height = in_height * 2;
    int out_width = in_width * 2;
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 512;
    int grid_size = (total_outputs + block_size - 1) / block_size;
    upsample2d_forward_kernel<<<grid_size, block_size>>>(
        dev_input_data, d_output, batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void conv2d_backwardev_input_data_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ dL_doutput,
    float* __restrict__ dL_dinput,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    const int kernel_size = 3;
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
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int oh = ih + pad - ky;
                int ow = iw + pad - kx;
                if (oh % stride == 0 && ow % stride == 0) {
                    oh /= stride;
                    ow /= stride;
                    if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                        int dout_idx = idx4(n, oc, oh, ow, out_channels, height, width);
                        int w_idx    = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                        acc += dL_doutput[dout_idx] * weights[w_idx];
                    }
                }
            }
        }
    }

    dL_dinput[tid] = acc;
}

__global__ void conv2d_backward_weights_kernel(
    const float* __restrict__ input,
    const float* __restrict__ dL_doutput,
    float* __restrict__ dL_dweights,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    const int kernel_size = 3;
    const int pad = 1;
    const int stride = 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_channels * in_channels * kernel_size * kernel_size;
    if (tid >= total) return;

    int tmp = tid;
    int kx = tmp % kernel_size; tmp /= kernel_size;
    int ky = tmp % kernel_size; tmp /= kernel_size;
    int ic = tmp % in_channels; tmp /= in_channels;
    int oc = tmp;

    float acc = 0.0f;

    for (int n = 0; n < batch_size; ++n) {
        for (int oh = 0; oh < height; ++oh) {
            for (int ow = 0; ow < width; ++ow) {
                int iy = oh * stride + ky - pad;
                int ix = ow * stride + kx - pad;
                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    int in_idx   = idx4(n, ic, iy, ix, in_channels, height, width);
                    int dout_idx = idx4(n, oc, oh, ow, out_channels, height, width);
                    acc += input[in_idx] * dL_doutput[dout_idx];
                }
            }
        }
    }

    int w_idx = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
    dL_dweights[w_idx] = acc;
}

__global__ void conv2d_backward_bias_kernel(
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
    for (int n = 0; n < batch_size; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = idx4(n, oc, h, w, out_channels, height, width);
                acc += dL_doutput[idx];
            }
        }
    }
    dL_dbias[oc] = acc;
}

void gpu_conv2d_backward(
    const float* dev_input_data, const float* d_weights, const float* d_dL_doutput,
    float* dev_grad_input, float* d_dL_dweights, float* d_dL_dbias,
    int batch_size, int in_channels, int out_channels, int height, int width
) {
    int block_size = 512;

    // dL/dinput
    if (dev_grad_input) {
        int total_inputs = batch_size * in_channels * height * width;
        int grid_size_input = (total_inputs + block_size - 1) / block_size;
        conv2d_backwardev_input_data_kernel<<<grid_size_input, block_size>>>(
            d_weights, d_dL_doutput, dev_grad_input, batch_size, in_channels,
            out_channels, height, width
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // dL/dweights
    if (d_dL_dweights) {
        int total_weights = out_channels * in_channels * 3 * 3;
        int grid_size_weights = (total_weights + block_size - 1) / block_size;
        conv2d_backward_weights_kernel<<<grid_size_weights, block_size>>>(
            dev_input_data, d_dL_doutput, d_dL_dweights, batch_size, in_channels,
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
    const float* d_output, const float* d_dL_doutput, float* dev_grad_input, int size
) {
    int block_size = 512;
    int grid_size = (size + block_size - 1) / block_size;
    relu_backward_kernel<<<grid_size, block_size>>>(
        d_output, d_dL_doutput, dev_grad_input, size
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
    const float* dev_input_data, const float* d_output, const float* d_dL_doutput,
    float* dev_grad_input, int batch_size, int channels, int in_height, int in_width
) {
    int out_height = in_height / 2;
    int out_width = in_width / 2;
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 512;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    // Khởi tạo dL_dinput bằng 0 trước khi sử dụng atomicAdd
    int input_size = batch_size * channels * in_height * in_width;
    CUDA_CHECK(cudaMemset(dev_grad_input, 0, input_size * sizeof(float)));

    maxpool2d_backward_kernel<<<grid_size, block_size>>>(
        dev_input_data, d_output, d_dL_doutput, dev_grad_input, batch_size, channels,
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
    const float* d_dL_doutput, float* dev_grad_input, int batch_size, int channels,
    int in_height, int in_width
) {
    int total_inputs = batch_size * channels * in_height * in_width;
    int block_size = 512;
    int grid_size = (total_inputs + block_size - 1) / block_size;
    upsample2d_backward_kernel<<<grid_size, block_size>>>(
        d_dL_doutput, dev_grad_input, batch_size, channels, in_height, in_width
    );
    CUDA_CHECK(cudaGetLastError());
}

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
    const float* d_output, const float* dev_target_data, float* d_dL_doutput, int size
) {
    int block_size = 512;
    int grid_size = (size + block_size - 1) / block_size;
    mse_loss_gradient_kernel<<<grid_size, block_size>>>(
        d_output, dev_target_data, d_dL_doutput, size
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

float gpu_mse_loss(const float* d_output, const float* dev_target_data, int size) {
    int block_size = 512;
    int num_blocks = (size + block_size - 1) / block_size;
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(float)));

    // 1. Compute partial sums
    mse_loss_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_output, dev_target_data, d_partial_sums, size
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
    int block_size = 512;
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



GPUAutoencoder::GPUAutoencoder() {
    // Host pointers
    host_enc_conv1_w = host_enc_conv1_b = host_enc_conv2_w = host_enc_conv2_b = nullptr; // Encoder Host weights
    host_dec_conv1_w = host_dec_conv1_b = host_dec_conv2_w = host_dec_conv2_b = host_dec_conv3_w = host_dec_conv3_b = nullptr; // Decoder Host weights

    // Device weight pointers
    dev_enc_conv1_w = dev_enc_conv1_b = dev_enc_conv2_w = dev_enc_conv2_b = nullptr; // Encoder device weights
    dev_dec_conv1_w = dev_dec_conv1_b = dev_dec_conv2_w = dev_dec_conv2_b = dev_dec_conv3_w = dev_dec_conv3_b = nullptr; // Decoder device  weights

    // Device gradient pointers
    dev_grad_enc_conv1_w = dev_grad_enc_conv1_b = dev_grad_enc_conv2_w = dev_grad_enc_conv2_b = nullptr; //Encoder device gradients

    dev_grad_dec_conv1_w = dev_grad_dec_conv1_b = dev_grad_dec_conv2_w = dev_grad_dec_conv2_b = dev_grad_dec_conv3_w = dev_grad_dec_conv3_b = nullptr; //Decoder device gradients

    // Device activation pointers
    dev_input_data = dev_target_data = nullptr; // Input and target data
    dev_enc_act1 = dev_enc_pool1 = dev_enc_act2 = dev_latent = nullptr; // Encoder activations
    dev_dec_conv1_out = dev_dec_upsample1 = dev_dec_act1 = dev_dec_upsample2 = dev_dec_out = nullptr; // Decoder activations

    // Device gradient buffers
    dev_grad_dec_out = dev_grad_dec_outdev_grad_dec_upsample2 = dev_grad_dec_act1 = dev_grad_dec_upsample1 = nullptr; 
    dev_grad_dec_conv1 = dev_grad_latent = dev_grad_enc_act2 = dev_grad_enc_pool1 = nullptr;
    dev_grad_enc_act1 = dev_grad_input = nullptr;

    batch_size = 0;
    max_batch_size = 0;
    memory_allocated = false;
}

GPUAutoencoder::~GPUAutoencoder() {
    free_device_memory();
    free_host_memory();
}

void GPUAutoencoder::allocate_host_memory() {
    if (host_enc_conv1_w) return; 

    host_enc_conv1_w = new float[W1_SIZE];
    host_enc_conv1_b = new float[B1_SIZE];
    host_enc_conv2_w = new float[W2_SIZE];
    host_enc_conv2_b = new float[B2_SIZE];
    host_dec_conv1_w = new float[W3_SIZE];
    host_dec_conv1_b = new float[B3_SIZE];
    host_dec_conv2_w = new float[W4_SIZE];
    host_dec_conv2_b = new float[B4_SIZE];
    host_dec_conv3_w = new float[W5_SIZE];
    host_dec_conv3_b = new float[B5_SIZE];
}

void GPUAutoencoder::free_host_memory() {
    delete[] host_enc_conv1_w; host_enc_conv1_w = nullptr;
    delete[] host_enc_conv1_b; host_enc_conv1_b = nullptr;
    delete[] host_enc_conv2_w; host_enc_conv2_w = nullptr;
    delete[] host_enc_conv2_b; host_enc_conv2_b = nullptr;
    delete[] host_dec_conv1_w; host_dec_conv1_w = nullptr;
    delete[] host_dec_conv1_b; host_dec_conv1_b = nullptr;
    delete[] host_dec_conv2_w; host_dec_conv2_w = nullptr;
    delete[] host_dec_conv2_b; host_dec_conv2_b = nullptr;
    delete[] host_dec_conv3_w; host_dec_conv3_w = nullptr;
    delete[] host_dec_conv3_b; host_dec_conv3_b = nullptr;
}

void GPUAutoencoder::allocate_device_memory(int requested_batch_size) {
    // Keep a sensible upper limit per allocation to avoid huge kernel launches.
    // We'll allocate buffers sized to requested_batch_size, but if already allocated
    // and large enough, we reuse them. If smaller, we reallocate.
    if (memory_allocated && requested_batch_size <= max_batch_size) {
        return;
    }

    if (memory_allocated) {
        free_device_memory();
        copy_weights_to_device();
    }

    max_batch_size = requested_batch_size;

    // Allocate device weights
    CUDA_CHECK(cudaMalloc(&dev_enc_conv1_w, W1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_enc_conv1_b, B1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_enc_conv2_w, W2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_enc_conv2_b, B2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dec_conv1_w, W3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dec_conv1_b, B3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dec_conv2_w, W4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dec_conv2_b, B4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dec_conv3_w, W5_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dec_conv3_b, B5_SIZE * sizeof(float)));

    // Allocate device gradients
    CUDA_CHECK(cudaMalloc(&dev_grad_enc_conv1_w, W1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_enc_conv1_b, B1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_enc_conv2_w, W2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_enc_conv2_b, B2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_conv1_w, W3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_conv1_b, B3_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_conv2_w, W4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_conv2_b, B4_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_conv3_w, W5_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_conv3_b, B5_SIZE * sizeof(float)));

    // Allocate device activations sized to max_batch_size
    CUDA_CHECK(cudaMalloc(&dev_input_data, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_target_data, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_enc_act1, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_enc_pool1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_enc_act2, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_latent, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dec_conv1_out, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dec_upsample1, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dec_act1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dec_upsample2, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dec_out, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_out, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_outdev_grad_dec_upsample2, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_act1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_upsample1, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_conv1, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_latent, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_enc_act2, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_enc_pool1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_enc_act1, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_input, max_batch_size * 3 * 32 * 32 * sizeof(float)));

    memory_allocated = true;
    
    // act4: batch x 256 x 16 x 16
    CUDA_CHECK(cudaMalloc(&dev_dec_act1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    
    // up2: batch x 256 x 32 x 32
    CUDA_CHECK(cudaMalloc(&dev_dec_upsample2, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    
    // act5 (output): batch x 3 x 32 x 32
    CUDA_CHECK(cudaMalloc(&dev_dec_out, max_batch_size * 3 * 32 * 32 * sizeof(float)));

    // Allocate gradient buffers for backward pass
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_out, max_batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_outdev_grad_dec_upsample2, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_act1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_upsample1, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_dec_conv1, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_latent, max_batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_enc_act2, max_batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_enc_pool1, max_batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_enc_act1, max_batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_grad_input, max_batch_size * 3 * 32 * 32 * sizeof(float)));

    memory_allocated = true;
}

void GPUAutoencoder::free_device_memory() {
    if (!memory_allocated) return;

    // Free device weights
    if (dev_enc_conv1_w) cudaFree(dev_enc_conv1_w); dev_enc_conv1_w = nullptr;
    if (dev_enc_conv1_b) cudaFree(dev_enc_conv1_b); dev_enc_conv1_b = nullptr;
    if (dev_enc_conv2_w) cudaFree(dev_enc_conv2_w); dev_enc_conv2_w = nullptr;
    if (dev_enc_conv2_b) cudaFree(dev_enc_conv2_b); dev_enc_conv2_b = nullptr;
    if (dev_dec_conv1_w) cudaFree(dev_dec_conv1_w); dev_dec_conv1_w = nullptr;
    if (dev_dec_conv1_b) cudaFree(dev_dec_conv1_b); dev_dec_conv1_b = nullptr;
    if (dev_dec_conv2_w) cudaFree(dev_dec_conv2_w); dev_dec_conv2_w = nullptr;
    if (dev_dec_conv2_b) cudaFree(dev_dec_conv2_b); dev_dec_conv2_b = nullptr;
    if (dev_dec_conv3_w) cudaFree(dev_dec_conv3_w); dev_dec_conv3_w = nullptr;
    if (dev_dec_conv3_b) cudaFree(dev_dec_conv3_b); dev_dec_conv3_b = nullptr;

    // Free device gradients
    if (dev_grad_enc_conv1_w) cudaFree(dev_grad_enc_conv1_w); dev_grad_enc_conv1_w = nullptr;
    if (dev_grad_enc_conv1_b) cudaFree(dev_grad_enc_conv1_b); dev_grad_enc_conv1_b = nullptr;
    if (dev_grad_enc_conv2_w) cudaFree(dev_grad_enc_conv2_w); dev_grad_enc_conv2_w = nullptr;
    if (dev_grad_enc_conv2_b) cudaFree(dev_grad_enc_conv2_b); dev_grad_enc_conv2_b = nullptr;
    if (dev_grad_dec_conv1_w) cudaFree(dev_grad_dec_conv1_w); dev_grad_dec_conv1_w = nullptr;
    if (dev_grad_dec_conv1_b) cudaFree(dev_grad_dec_conv1_b); dev_grad_dec_conv1_b = nullptr;
    if (dev_grad_dec_conv2_w) cudaFree(dev_grad_dec_conv2_w); dev_grad_dec_conv2_w = nullptr;
    if (dev_grad_dec_conv2_b) cudaFree(dev_grad_dec_conv2_b); dev_grad_dec_conv2_b = nullptr;
    if (dev_grad_dec_conv3_w) cudaFree(dev_grad_dec_conv3_w); dev_grad_dec_conv3_w = nullptr;
    if (dev_grad_dec_conv3_b) cudaFree(dev_grad_dec_conv3_b); dev_grad_dec_conv3_b = nullptr;

    // Free device activations
    if (dev_input_data) cudaFree(dev_input_data); dev_input_data = nullptr;
    if (dev_target_data) cudaFree(dev_target_data); dev_target_data = nullptr;
    if (dev_enc_act1) cudaFree(dev_enc_act1); dev_enc_act1 = nullptr;
    if (dev_enc_pool1) cudaFree(dev_enc_pool1); dev_enc_pool1 = nullptr;
    if (dev_enc_act2) cudaFree(dev_enc_act2); dev_enc_act2 = nullptr;
    if (dev_latent) cudaFree(dev_latent); dev_latent = nullptr;
    if (dev_dec_conv1_out) cudaFree(dev_dec_conv1_out); dev_dec_conv1_out = nullptr;
    if (dev_dec_upsample1) cudaFree(dev_dec_upsample1); dev_dec_upsample1 = nullptr;
    if (dev_dec_act1) cudaFree(dev_dec_act1); dev_dec_act1 = nullptr;
    if (dev_dec_upsample2) cudaFree(dev_dec_upsample2); dev_dec_upsample2 = nullptr;
    if (dev_dec_out) cudaFree(dev_dec_out); dev_dec_out = nullptr;

    // Free gradient buffers
    if (dev_grad_dec_out) cudaFree(dev_grad_dec_out); dev_grad_dec_out = nullptr;
    if (dev_grad_dec_outdev_grad_dec_upsample2) cudaFree(dev_grad_dec_outdev_grad_dec_upsample2); dev_grad_dec_outdev_grad_dec_upsample2 = nullptr;
    if (dev_grad_dec_act1) cudaFree(dev_grad_dec_act1); dev_grad_dec_act1 = nullptr;
    if (dev_grad_dec_upsample1) cudaFree(dev_grad_dec_upsample1); dev_grad_dec_upsample1 = nullptr;
    if (dev_grad_dec_conv1) cudaFree(dev_grad_dec_conv1); dev_grad_dec_conv1 = nullptr;
    if (dev_grad_latent) cudaFree(dev_grad_latent); dev_grad_latent = nullptr;
    if (dev_grad_enc_act2) cudaFree(dev_grad_enc_act2); dev_grad_enc_act2 = nullptr;
    if (dev_grad_enc_pool1) cudaFree(dev_grad_enc_pool1); dev_grad_enc_pool1 = nullptr;
    if (dev_grad_enc_act1) cudaFree(dev_grad_enc_act1); dev_grad_enc_act1 = nullptr;
    if (dev_grad_input) cudaFree(dev_grad_input); dev_grad_input = nullptr;

    memory_allocated = false;
}

void GPUAutoencoder::initialize() {
    allocate_host_memory();

    // Initialize weights using Xavier initialization
    init_weights_xavier(host_enc_conv1_w, 3, 256);
    init_weights_xavier(host_enc_conv2_w, 256, 128);
    init_weights_xavier(host_dec_conv1_w, 128, 128);
    init_weights_xavier(host_dec_conv2_w, 128, 256);
    init_weights_xavier(host_dec_conv3_w, 256, 3);

    // Initialize biases to zero
    memset(host_enc_conv1_b, 0, B1_SIZE * sizeof(float));
    memset(host_enc_conv2_b, 0, B2_SIZE * sizeof(float));
    memset(host_dec_conv1_b, 0, B3_SIZE * sizeof(float));
    memset(host_dec_conv2_b, 0, B4_SIZE * sizeof(float));
    memset(host_dec_conv3_b, 0, B5_SIZE * sizeof(float));

    // Allocate device memory and copy weights
    allocate_device_memory(64);
    copy_weights_to_device();
}

void GPUAutoencoder::forward_device(const float* d_in, int batch_size) {
    batch_size = batch_size;

    // Encoder
    // Conv1: 3->256, 32x32 + ReLU
    gpu_conv2d_forward(d_in, dev_enc_conv1_w, dev_enc_conv1_b, dev_enc_act1, batch_size, 3, 256, 32, 32);
    gpu_relu_forward(dev_enc_act1, batch_size * 256 * 32 * 32);

    // MaxPool1: 32x32->16x16
    gpu_maxpool2d_forward(dev_enc_act1, dev_enc_pool1, batch_size, 256, 32, 32);

    // Conv2: 256->128, 16x16 + ReLU
    gpu_conv2d_forward(dev_enc_pool1, dev_enc_conv2_w, dev_enc_conv2_b, dev_enc_act2, batch_size, 256, 128, 16, 16);
    gpu_relu_forward(dev_enc_act2, batch_size * 128 * 16 * 16);

    // MaxPool2 (encoded layer): 16x16->8x8
    gpu_maxpool2d_forward(dev_enc_act2, dev_latent, batch_size, 128, 16, 16);

    // Decoder
    // Conv3: 128->128, 8x8 + ReLU
    gpu_conv2d_forward(dev_latent, dev_dec_conv1_w, dev_dec_conv1_b, dev_dec_conv1_out, batch_size, 128, 128, 8, 8);
    gpu_relu_forward(dev_dec_conv1_out, batch_size * 128 * 8 * 8);

    // Upsample1: 8x8->16x16
    gpu_upsample2d_forward(dev_dec_conv1_out, dev_dec_upsample1, batch_size, 128, 8, 8);

    // Conv4: 128->256, 16x16 + ReLU
    gpu_conv2d_forward(dev_dec_upsample1, dev_dec_conv2_w, dev_dec_conv2_b, dev_dec_act1, batch_size, 128, 256, 16, 16);
    gpu_relu_forward(dev_dec_act1, batch_size * 256 * 16 * 16);

    // Upsample2: 16x16->32x32
    gpu_upsample2d_forward(dev_dec_act1, dev_dec_upsample2, batch_size, 256, 16, 16);

    // Conv5: 256->3, 32x32 (output, no activation)
    gpu_conv2d_forward(dev_dec_upsample2, dev_dec_conv3_w, dev_dec_conv3_b, dev_dec_out, batch_size, 256, 3, 32, 32);
}

void GPUAutoencoder::forward(const float* h_input, float* h_output, int batch_size) {
    // Device memory is allocated
    allocate_device_memory(batch_size);
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(dev_input_data, h_input, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));

    // Run forward pass on device
    forward_device(dev_input_data, batch_size);

    // Copy output back to host
    CUDA_CHECK(cudaMemcpy(h_output, dev_dec_out, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPUAutoencoder::backward_device(const float* d_in, const float* d_tgt, int batch_size) {
    int output_size = batch_size * 3 * 32 * 32;

    // 1. Compute gradient at output: dL/dact5 = 2(act5 - target) / N
    gpu_mse_loss_gradient(dev_dec_out, d_tgt, dev_grad_dec_out, output_size);

    // 2. Backward through Conv5: 256->3, 32x32
    // dev_grad_input cho Conv5 là dev_grad_dec_outdev_grad_dec_upsample2
    gpu_conv2d_backward(dev_dec_upsample2, dev_dec_conv3_w, dev_grad_dec_out, dev_grad_dec_outdev_grad_dec_upsample2, dev_grad_dec_conv3_w, dev_grad_dec_conv3_b,
                        batch_size, 256, 3, 32, 32);

    // 3. Backward through Upsample2: 16x16->32x32
    // dev_grad_input cho Upsample2 là dev_grad_dec_act1
    gpu_upsample2d_backward(dev_grad_dec_outdev_grad_dec_upsample2, dev_grad_dec_act1, batch_size, 256, 16, 16);

    // 4. Backward through ReLU4
    gpu_relu_backward(dev_dec_act1, dev_grad_dec_act1, dev_grad_dec_act1, batch_size * 256 * 16 * 16);

    // 5. Backward through Conv4: 128->256, 16x16
    // dev_grad_input cho Conv4 là dev_grad_dec_upsample1
    gpu_conv2d_backward(dev_dec_upsample1, dev_dec_conv2_w, dev_grad_dec_act1, dev_grad_dec_upsample1, dev_grad_dec_conv2_w, dev_grad_dec_conv2_b,
                        batch_size, 128, 256, 16, 16);

    // 6. Backward through Upsample1: 8x8->16x16
    // dev_grad_input cho Upsample1 là dev_grad_dec_conv1
    gpu_upsample2d_backward(dev_grad_dec_upsample1, dev_grad_dec_conv1, batch_size, 128, 8, 8);

    // 7. Backward through ReLU3
    gpu_relu_backward(dev_dec_conv1_out, dev_grad_dec_conv1, dev_grad_dec_conv1, batch_size * 128 * 8 * 8);

    // 8. Backward through Conv3: 128->128, 8x8
    // dev_grad_input cho Conv3 là dev_grad_latent
    gpu_conv2d_backward(dev_latent, dev_dec_conv1_w, dev_grad_dec_conv1, dev_grad_latent, dev_grad_dec_conv1_w, dev_grad_dec_conv1_b,
                        batch_size, 128, 128, 8, 8);

    // 9. Backward through MaxPool2: 16x16->8x8
    // dev_grad_input cho MaxPool2 là dev_grad_enc_act2
    gpu_maxpool2d_backward(dev_enc_act2, dev_latent, dev_grad_latent, dev_grad_enc_act2,
                           batch_size, 128, 16, 16);

    // 10. Backward through ReLU2
    gpu_relu_backward(dev_enc_act2, dev_grad_enc_act2, dev_grad_enc_act2, batch_size * 128 * 16 * 16);

    // 11. Backward through Conv2: 256->128, 16x16
    // dev_grad_input cho Conv2 là dev_grad_enc_pool1
    gpu_conv2d_backward(dev_enc_pool1, dev_enc_conv2_w, dev_grad_enc_act2, dev_grad_enc_pool1, dev_grad_enc_conv2_w, dev_grad_enc_conv2_b,
                        batch_size, 256, 128, 16, 16);

    // 12. Backward through MaxPool1: 32x32->16x16
    // dev_grad_input cho MaxPool1 là dev_grad_enc_act1
    gpu_maxpool2d_backward(dev_enc_act1, dev_enc_pool1, dev_grad_enc_pool1, dev_grad_enc_act1,
                           batch_size, 256, 32, 32);

    // 13. Backward through ReLU1
    gpu_relu_backward(dev_enc_act1, dev_grad_enc_act1, dev_grad_enc_act1, batch_size * 256 * 32 * 32);

    // 14. Backward through Conv1: 3->256, 32x32
    // dev_grad_input cho Conv1 là dev_grad_input
    gpu_conv2d_backward(d_in, dev_enc_conv1_w, dev_grad_enc_act1, dev_grad_input, dev_grad_enc_conv1_w, dev_grad_enc_conv1_b,
                        batch_size, 3, 256, 32, 32);
}

void GPUAutoencoder::backward(const float* h_input, const float* h_target, int batch_size) {
    // Ensure device memory is allocated
    allocate_device_memory(batch_size);
    
    // Copy input and target to device
    CUDA_CHECK(cudaMemcpy(dev_input_data, h_input, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_target_data, h_target, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));

    // Run forward pass
    forward_device(dev_input_data, batch_size);

    // Run backward pass on device
    backward_device(dev_input_data, dev_target_data, batch_size);
}

void GPUAutoencoder::update_weights(float learning_rate) {
    const float clip_value = 1.0f;

    // Update all weights and biases using SGD with gradient clipping
    gpu_sgd_update(dev_enc_conv1_w, dev_grad_enc_conv1_w, learning_rate, clip_value, W1_SIZE);
    gpu_sgd_update(dev_enc_conv1_b, dev_grad_enc_conv1_b, learning_rate, clip_value, B1_SIZE);
    gpu_sgd_update(dev_enc_conv2_w, dev_grad_enc_conv2_w, learning_rate, clip_value, W2_SIZE);
    gpu_sgd_update(dev_enc_conv2_b, dev_grad_enc_conv2_b, learning_rate, clip_value, B2_SIZE);
    gpu_sgd_update(dev_dec_conv1_w, dev_grad_dec_conv1_w, learning_rate, clip_value, W3_SIZE);
    gpu_sgd_update(dev_dec_conv1_b, dev_grad_dec_conv1_b, learning_rate, clip_value, B3_SIZE);
    gpu_sgd_update(dev_dec_conv2_w, dev_grad_dec_conv2_w, learning_rate, clip_value, W4_SIZE);
    gpu_sgd_update(dev_dec_conv2_b, dev_grad_dec_conv2_b, learning_rate, clip_value, B4_SIZE);
    gpu_sgd_update(dev_dec_conv3_w, dev_grad_dec_conv3_w, learning_rate, clip_value, W5_SIZE);
    gpu_sgd_update(dev_dec_conv3_b, dev_grad_dec_conv3_b, learning_rate, clip_value, B5_SIZE);
}

float GPUAutoencoder::compute_loss(const float* h_target, int batch_size) {
    // Copy target to device
    CUDA_CHECK(cudaMemcpy(dev_target_data, h_target, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
    
    int size = batch_size * 3 * 32 * 32;
    return gpu_mse_loss(dev_dec_out, dev_target_data, size);
}

void GPUAutoencoder::extract_features_device(const float* d_in, float* d_features, int batch_size) {
    // Run encoder only
    // Conv1: 3->256, 32x32 + ReLU
    gpu_conv2d_forward(d_in, dev_enc_conv1_w, dev_enc_conv1_b, dev_enc_act1, batch_size, 3, 256, 32, 32);
    gpu_relu_forward(dev_enc_act1, batch_size * 256 * 32 * 32);

    // MaxPool1: 32x32->16x16
    gpu_maxpool2d_forward(dev_enc_act1, dev_enc_pool1, batch_size, 256, 32, 32);

    // Conv2: 256->128, 16x16 + ReLU
    gpu_conv2d_forward(dev_enc_pool1, dev_enc_conv2_w, dev_enc_conv2_b, dev_enc_act2, batch_size, 256, 128, 16, 16);
    gpu_relu_forward(dev_enc_act2, batch_size * 128 * 16 * 16);

    // MaxPool2 (encoded layer): 16x16->8x8
    // Gán d_features = dev_latent (output của MaxPool2)
    gpu_maxpool2d_forward(dev_enc_act2, d_features, batch_size, 128, 16, 16);
}

void GPUAutoencoder::extract_features(const float* h_input, float* h_features, int batch_size) {
    // Process large batch sizes in chunks to avoid launching kernels with
    // grid dimensions larger than the device limit.
    const int IMG_SZ = 3 * 32 * 32;
    const int FEATURE_SZ = 128 * 8 * 8; // 8192
    const int MAX_CHUNK = 1024; // reasonable chunk size per GPU launch

    int remaining = batch_size;
    int offset = 0;

    while (remaining > 0) {
        int chunk = std::min(remaining, MAX_CHUNK);

        // Ensure device buffers are allocated for the chunk size
        allocate_device_memory(chunk);

        // Copy this chunk of input to device
        CUDA_CHECK(cudaMemcpy(dev_input_data, h_input + offset * IMG_SZ, chunk * IMG_SZ * sizeof(float), cudaMemcpyHostToDevice));

        // Run encoder for this chunk
        extract_features_device(dev_input_data, dev_latent, chunk);

        // Copy features back to host
        CUDA_CHECK(cudaMemcpy(h_features + offset * FEATURE_SZ, dev_latent, chunk * FEATURE_SZ * sizeof(float), cudaMemcpyDeviceToHost));

        remaining -= chunk;
        offset += chunk;
    }
}


void GPUAutoencoder::copy_weights_to_device() {
    CUDA_CHECK(cudaMemcpy(dev_enc_conv1_w, host_enc_conv1_w, W1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_enc_conv1_b, host_enc_conv1_b, B1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_enc_conv2_w, host_enc_conv2_w, W2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_enc_conv2_b, host_enc_conv2_b, B2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_dec_conv1_w, host_dec_conv1_w, W3_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_dec_conv1_b, host_dec_conv1_b, B3_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_dec_conv2_w, host_dec_conv2_w, W4_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_dec_conv2_b, host_dec_conv2_b, B4_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_dec_conv3_w, host_dec_conv3_w, W5_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_dec_conv3_b, host_dec_conv3_b, B5_SIZE * sizeof(float), cudaMemcpyHostToDevice));
}

void GPUAutoencoder::copy_weights_to_host() {
    CUDA_CHECK(cudaMemcpy(host_enc_conv1_w, dev_enc_conv1_w, W1_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_enc_conv1_b, dev_enc_conv1_b, B1_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_enc_conv2_w, dev_enc_conv2_w, W2_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_enc_conv2_b, dev_enc_conv2_b, B2_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dec_conv1_w, dev_dec_conv1_w, W3_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dec_conv1_b, dev_dec_conv1_b, B3_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dec_conv2_w, dev_dec_conv2_w, W4_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dec_conv2_b, dev_dec_conv2_b, B4_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dec_conv3_w, dev_dec_conv3_w, W5_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dec_conv3_b, dev_dec_conv3_b, B5_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPUAutoencoder::save_weights(const std::string& filepath) {
    // Copy weights from device to host
    copy_weights_to_host();

    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filepath.c_str());
        return;
    }

    fwrite(host_enc_conv1_w, sizeof(float), W1_SIZE, f);
    fwrite(host_enc_conv1_b, sizeof(float), B1_SIZE, f);
    fwrite(host_enc_conv2_w, sizeof(float), W2_SIZE, f);
    fwrite(host_enc_conv2_b, sizeof(float), B2_SIZE, f);
    fwrite(host_dec_conv1_w, sizeof(float), W3_SIZE, f);
    fwrite(host_dec_conv1_b, sizeof(float), B3_SIZE, f);
    fwrite(host_dec_conv2_w, sizeof(float), W4_SIZE, f);
    fwrite(host_dec_conv2_b, sizeof(float), B4_SIZE, f);
    fwrite(host_dec_conv3_w, sizeof(float), W5_SIZE, f);
    fwrite(host_dec_conv3_b, sizeof(float), B5_SIZE, f);

    fclose(f);
    printf("GPU Model weights saved to: %s\n", filepath.c_str());
}

void GPUAutoencoder::load_weights(const std::string& filepath) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file for reading: %s\n", filepath.c_str());
        return;
    }

    if (!host_enc_conv1_w) {
        allocate_host_memory();
    }

    fread(host_enc_conv1_w, sizeof(float), W1_SIZE, f);
    fread(host_enc_conv1_b, sizeof(float), B1_SIZE, f);
    fread(host_enc_conv2_w, sizeof(float), W2_SIZE, f);
    fread(host_enc_conv2_b, sizeof(float), B2_SIZE, f);
    fread(host_dec_conv1_w, sizeof(float), W3_SIZE, f);
    fread(host_dec_conv1_b, sizeof(float), B3_SIZE, f);
    fread(host_dec_conv2_w, sizeof(float), W4_SIZE, f);
    fread(host_dec_conv2_b, sizeof(float), B4_SIZE, f);
    fread(host_dec_conv3_w, sizeof(float), W5_SIZE, f);
    fread(host_dec_conv3_b, sizeof(float), B5_SIZE, f);

    fclose(f);

    // Allocate device memory if needed and copy weights
    if (!memory_allocated) {
        allocate_device_memory(max_batch_size);
    }
    copy_weights_to_device();

    printf("GPU Model weights loaded from: %s\n", filepath.c_str());
}


