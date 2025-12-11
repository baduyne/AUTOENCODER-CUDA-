#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include"utils.h"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Network configuration (as specified)

constexpr int IMG_W = 32;
constexpr int IMG_H = 32;
constexpr int IMG_C = 3;

// Encoder/Decoder channels
constexpr int ENC1_OUT = 256; // first conv filters
constexpr int ENC2_OUT = 128; // second conv filters -> latent channels

// Latent dims
constexpr int LAT_W = 8;
constexpr int LAT_H = 8;
constexpr int LAT_C = 128; // equals ENC2_OUT

// Batch settings
int BATCH = 64; // can be changed at runtime

// Utility: random initialization for weight 
float randn(float mean=0.0f, float std=0.05f) {
    static std::mt19937 rng(12345);
    static std::normal_distribution<float> d(mean, std);
    return d(rng);
}

// Index helpers for NCHW layout
__host__ __device__ inline int idx4(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

// convolution kernel (NCHW, filters: out_channels x in_channels x K x K)
// Each thread computes one output pixel (n, out_c, out_h, out_w)
__global__ void conv2d_naive(const float* __restrict__ input, int N, int inC, int inH, int inW,
                             const float* __restrict__ filters, int outC, int K,
                             int pad, int stride, float* __restrict__ output, int outH, int outW)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * outC * outH * outW;
    if (tid >= total) return;

    // decode indices
    int tmp = tid;
    int ow = tmp % outW; tmp /= outW;
    int oh = tmp % outH; tmp /= outH;
    int oc = tmp % outC; tmp /= outC;
    int n  = tmp;

    float acc = 0.0f;
    for (int ic = 0; ic < inC; ++ic) {
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                int in_y = oh * stride + ky - pad;
                int in_x = ow * stride + kx - pad;
                if (in_y >= 0 && in_y < inH && in_x >= 0 && in_x < inW) {
                    int in_idx = idx4(n, ic, in_y, in_x, inC, inH, inW);
                    // filter layout: oc, ic, ky, kx -> linear index
                    int f_idx = ((oc * inC + ic) * K + ky) * K + kx;
                    acc += input[in_idx] * filters[f_idx];
                }
            }
        }
    }
    int out_idx = idx4(n, oc, oh, ow, outC, outH, outW);
    output[out_idx] = acc;
}

// Add bias kernel (per-channel bias)
__global__ void add_bias(float* __restrict__ output, const float* __restrict__ bias, int N, int C, int H, int W)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (tid >= total) return;
    int tmp = tid;
    int w = tmp % W; tmp /= W;
    int h = tmp % H; tmp /= H;
    int c = tmp % C; tmp /= C;
    int n = tmp;
    int out_idx = idx4(n, c, h, w, C, H, W);
    output[out_idx] += bias[c];
}

// ReLU inplace
__global__ void relu_inplace(float* __restrict__ x, int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    float v = x[tid];

    // hàm relu max(x,0)
    x[tid] = v > 0.0f ? v : 0.0f;
}

// MaxPool 2x2 stride 2 (NCHW), output dims floor/ceil chosen by caller
__global__ void maxpool2x2(const float* __restrict__ input, int N, int C, int inH, int inW,
                           float* __restrict__ output, int outH, int outW)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (tid >= total) return;
    int tmp = tid;
    int ow = tmp % outW; tmp /= outW;
    int oh = tmp % outH; tmp /= outH;
    int c = tmp % C; tmp /= C;
    int n = tmp;

    int in_y = oh * 2;
    int in_x = ow * 2;

    float best = -1e30f;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int y = in_y + dy;
            int x = in_x + dx;
            if (y < inH && x < inW) {
                int in_idx = idx4(n, c, y, x, C, inH, inW);
                float v = input[in_idx];
                if (v > best) best = v;
            }
        }
    }
    int out_idx = idx4(n, c, oh, ow, C, outH, outW);
    output[out_idx] = best;
}

// Upsample nearest neighbor 2x
__global__ void upsample2x(const float* __restrict__ input, int N, int C, int inH, int inW,
                           float* __restrict__ output, int outH, int outW)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (tid >= total) return;
    int tmp = tid;
    int ow = tmp % outW; tmp /= outW;
    int oh = tmp % outH; tmp /= outH;
    int c = tmp % C; tmp /= C;
    int n = tmp;

    int in_y = oh / 2;
    int in_x = ow / 2;
    if (in_y >= inH) in_y = inH - 1;
    if (in_x >= inW) in_x = inW - 1;
    int in_idx = idx4(n, c, in_y, in_x, C, inH, inW);
    int out_idx = idx4(n, c, oh, ow, C, outH, outW);
    output[out_idx] = input[in_idx];
}

// MSE loss kernel: each thread accumulates per-element squared error into a global accumulator using atomicAdd.
__global__ void mse_loss_and_grad(const float* __restrict__ output, const float* __restrict__ target,
                                  int total, float* __restrict__ loss_accum, float* __restrict__ grad)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    float diff = output[tid] - target[tid];
    atomicAdd(loss_accum, diff * diff);
    grad[tid] = 2.0f * diff; // gradient wrt output
}
__global__ void relu_backward(const float* __restrict__ out,
                              const float* __restrict__ dOut,
                              float* __restrict__ dIn,
                              int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    dIn[tid] = (out[tid] > 0.0f) ? dOut[tid] : 0.0f;
}
__global__ void maxpool2x2_backward(
    const float* __restrict__ input,   // trước pool
    const float* __restrict__ output,  // sau pool
    const float* __restrict__ dOut,    // gradient sau pool
    float* __restrict__ dIn,           // gradient trước pool
    int N, int C, int inH, int inW, int outH, int outW)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (tid >= total) return;

    int tmp = tid;
    int ow = tmp % outW; tmp /= outW;
    int oh = tmp % outH; tmp /= outH;
    int c  = tmp % C;    tmp /= C;
    int n  = tmp;

    float pooled = output[tid];
    float g = dOut[tid];

    int base_y = oh * 2;
    int base_x = ow * 2;

    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int y = base_y + dy;
            int x = base_x + dx;
            if (y < inH && x < inW) {
                int idx = idx4(n, c, y, x, C, inH, inW);
                if (input[idx] == pooled)
                    atomicAdd(&dIn[idx], g);
            }
        }
    }
}


__global__ void upsample2x_backward(
    const float* __restrict__ dOut, // gradient after upsample
    float* __restrict__ dIn,        // gradient before upsample
    int N, int C, int inH, int inW)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * inH * inW;
    if (tid >= total) return;

    int idx = tid;

    int tmp = tid;
    int w = tmp % inW; tmp /= inW;
    int h = tmp % inH; tmp /= inH;
    int c = tmp % C;   tmp /= C;
    int n = tmp;

    int outH = inH * 2;
    int outW = inW * 2;

    float acc = 0.0f;

    // corresponding 2x2 block in upsampled image
    int y0 = h * 2;
    int x0 = w * 2;

    for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++)
        {
            int y = y0 + dy;
            int x = x0 + dx;
            int out_idx = idx4(n, c, y, x, C, outH, outW);
            acc += dOut[out_idx];
        }

    dIn[idx] = acc;
}
__global__ void conv2d_grad_input_naive(
    const float* __restrict__ dOut,
    const float* __restrict__ filters,
    float* __restrict__ dIn,
    int N,
    int inC, int inH, int inW,
    int outC, int outH, int outW,
    int K, int pad, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * inC * inH * inW;
    if (tid >= total) return;

    int tmp = tid;
    int w = tmp % inW;  tmp /= inW;
    int h = tmp % inH;  tmp /= inH;
    int c = tmp % inC;  tmp /= inC;
    int n = tmp;

    float acc = 0.0f;

    for (int oc = 0; oc < outC; oc++) {
        for (int ky = 0; ky < K; ky++) {
            for (int kx = 0; kx < K; kx++) {

                int oh = (h + pad - ky);
                int ow = (w + pad - kx);

                if (oh % stride == 0 && ow % stride == 0) {
                    oh /= stride;
                    ow /= stride;

                    if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                        int dOut_idx = idx4(n, oc, oh, ow, outC, outH, outW);
                        int fidx = ((oc * inC + c) * K + ky) * K + kx;
                        acc += dOut[dOut_idx] * filters[fidx];
                    }
                }
            }
        }
    }

    dIn[tid] = acc;
}


// Zero array
__global__ void zero_kernel(float* ptr, int n) { int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid < n) ptr[tid]=0.0f; }

// Compute gradient w.r.t conv filters naively
// For each filter element (oc, ic, ky, kx) we accumulate over batch and spatial positions
__global__ void conv2d_grad_filters_naive(const float* __restrict__ input, const float* __restrict__ dOut,
                                          int N, int inC, int inH, int inW,
                                          int outC, int outH, int outW, int K, int pad, int stride,
                                          float* __restrict__ gradFilters)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outC * inC * K * K;
    if (tid >= total) return;

    int tmp = tid;
    int kx = tmp % K; tmp /= K;
    int ky = tmp % K; tmp /= K;
    int ic = tmp % inC; tmp /= inC;
    int oc = tmp;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                int in_y = oh * stride + ky - pad;
                int in_x = ow * stride + kx - pad;
                if (in_y >= 0 && in_y < inH && in_x >= 0 && in_x < inW) {
                    int in_idx = idx4(n, ic, in_y, in_x, inC, inH, inW);
                    int dout_idx = idx4(n, oc, oh, ow, outC, outH, outW);
                    acc += input[in_idx] * dOut[dout_idx];
                }
            }
        }
    }
    int gidx = ((oc * inC + ic) * K + ky) * K + kx;
    gradFilters[gidx] = acc;
}

// Compute gradient w.r.t bias (sum over N,H,W for each channel)
__global__ void conv2d_grad_bias_naive(const float* __restrict__ dOut, int N, int outC, int outH, int outW, float* __restrict__ gradBias)
{
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= outC) return;
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < outH; ++h) for (int w = 0; w < outW; ++w) {
            int idx = idx4(n, oc, h, w, outC, outH, outW);
            acc += dOut[idx];
        }
    }
    gradBias[oc] = acc;
}

__global__ void conv2d_grad_bias(
    const float* __restrict__ dOut,
    float* __restrict__ dB,
    int N, int C, int H, int W)
{
    int oc = blockIdx.x;
    float acc = 0.f;

    for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
            {
                int idx = ((n*C + oc) * H + h) * W + w;
                acc += dOut[idx];
            }
    atomicAdd(&dB[oc], acc);
}


__global__ void update_weights_kernel(
    float* W,       // weights
    float* dW,      // gradient of weights
    float  lr,      
    int n           // number of weight elements
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        W[idx] -= lr * dW[idx];
        dW[idx] = 0.0f;   // reset gradient
    }
}

__global__ void update_bias_kernel(
    float* B, 
    float* dB,
    float lr,
    int n
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        B[idx] -= lr * dB[idx];
        dB[idx] = 0.0f;
    }
}

void update_weights(
    float* W, float* B,
    float* dW, float* dB,
    float lr,
    int weight_count,
    int bias_count
){
    int block = 256;
    int gridW = (weight_count + block - 1) / block;
    int gridB = (bias_count + block - 1) / block;

    update_weights_kernel<<<gridW, block>>>(W, dW, lr, weight_count);
    update_bias_kernel<<<gridB, block>>>(B, dB, lr, bias_count);
}


// Update parameters: w -= lr * grad
__global__ void sgd_update(float* __restrict__ params, const float* __restrict__ grads, int n, float lr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    params[tid] -= lr * grads[tid];
}

// GPUAutoencoder class: manages allocations and forward/backward pipelines
struct GPUAutoencoder {
    // ============================
    // WEIGHTS
    // ============================
    float *d_w_e1, *d_b_e1;     // encoder 1
    float *d_w_e2, *d_b_e2;     // encoder 2

    float *d_w_d1, *d_b_d1;     // decoder 1
    float *d_w_d2, *d_b_d2;     // decoder 2
    float *d_w_out, *d_b_out;   // final conv

    // ============================
    // ACTIVATIONS (forward)
    // ============================
    float *d_input;

    float *d_e1;
    float *d_p1;

    float *d_e2;
    float *d_p2;

    float *d_dec1;
    float *d_up1;

    float *d_dec2;
    float *d_up2;

    float *d_out;

    // ============================
    // GRADIENTS (activations)
    // ============================
    float *d_d_out;
    float *d_d_up2;
    float *d_d_dec2;
    float *d_d_up1;
    float *d_d_dec1;
    float *d_d_p2;
    float *d_d_e2;
    float *d_d_p1;
    float *d_d_e1;
    float *d_d_input;

    // ============================
    // GRADIENTS (weights)
    // ============================
    float *d_dw_e1, *d_db_e1;
    float *d_dw_e2, *d_db_e2;

    float *d_dw_d1, *d_db_d1;
    float *d_dw_d2, *d_db_d2;

    float *d_dw_out, *d_db_out;

    // other
    float *d_loss_accum;
    float *d_out_grad; // gradient wrt ou

    int batch_size;

    GPUAutoencoder(int batch_ = 64) : batch_size(batch_) {
        BATCH = batch_;
        allocate();
        init_params();
    }

    void allocate() {
        int B = batch_size;

        cudaMalloc(&d_w_e1,   (size_t)ENC1_OUT * IMG_C   * 3 * 3 * sizeof(float));
        cudaMalloc(&d_b_e1,   (size_t)ENC1_OUT * sizeof(float));

        cudaMalloc(&d_w_e2,   (size_t)ENC2_OUT * ENC1_OUT * 3 * 3 * sizeof(float));
        cudaMalloc(&d_b_e2,   (size_t)ENC2_OUT * sizeof(float));

        cudaMalloc(&d_w_d1,   (size_t)ENC2_OUT * ENC2_OUT * 3 * 3 * sizeof(float));
        cudaMalloc(&d_b_d1,   (size_t)ENC2_OUT * sizeof(float));

        cudaMalloc(&d_w_d2,   (size_t)ENC1_OUT * ENC2_OUT * 3 * 3 * sizeof(float));
        cudaMalloc(&d_b_d2,   (size_t)ENC1_OUT * sizeof(float));

        cudaMalloc(&d_w_out,  (size_t)IMG_C    * ENC1_OUT * 3 * 3 * sizeof(float));
        cudaMalloc(&d_b_out,  (size_t)IMG_C * sizeof(float));

        // ---------------------------
        // Forward activations
        // ---------------------------
        cudaMalloc(&d_input,   B * 3     * 32 * 32 * sizeof(float));
        cudaMalloc(&d_e1,      B * 256   * 32 * 32 * sizeof(float));
        cudaMalloc(&d_p1,      B * 256   * 16 * 16 * sizeof(float));
        cudaMalloc(&d_e2,      B * 128   * 16 * 16 * sizeof(float));
        cudaMalloc(&d_p2,      B * 128   *  8 *  8 * sizeof(float));

        cudaMalloc(&d_dec1,    B * 128   *  8 *  8 * sizeof(float));
        cudaMalloc(&d_up1,     B * 128   * 16 * 16 * sizeof(float));
        cudaMalloc(&d_dec2,    B * 256   * 16 * 16 * sizeof(float));
        cudaMalloc(&d_up2,     B * 256   * 32 * 32 * sizeof(float));
        cudaMalloc(&d_out,     B * 3     * 32 * 32 * sizeof(float));

        // ---------------------------
        // Gradients of activations
        // ---------------------------
        cudaMalloc(&d_d_out,   B * 3     * 32 * 32 * sizeof(float));
        cudaMalloc(&d_d_up2,   B * 256   * 32 * 32 * sizeof(float));
        cudaMalloc(&d_d_dec2,  B * 256   * 16 * 16 * sizeof(float));
        cudaMalloc(&d_d_up1,   B * 128   * 16 * 16 * sizeof(float));
        cudaMalloc(&d_d_dec1,  B * 128   *  8 *  8 * sizeof(float));

        cudaMalloc(&d_d_p2,    B * 128   *  8 *  8 * sizeof(float));
        cudaMalloc(&d_d_e2,    B * 128   * 16 * 16 * sizeof(float));
        cudaMalloc(&d_d_p1,    B * 256   * 16 * 16 * sizeof(float));
        cudaMalloc(&d_d_e1,    B * 256   * 32 * 32 * sizeof(float));

        cudaMalloc(&d_d_input, B * 3     * 32 * 32 * sizeof(float));

        // ---------------------------
        // Weight gradients
        // ---------------------------
        cudaMalloc(&d_dw_e1,   256 * 3   * 3 * 3 * sizeof(float));
        cudaMalloc(&d_db_e1,   256 * sizeof(float));

        cudaMalloc(&d_dw_e2,   128 * 256 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_db_e2,   128 * sizeof(float));

        cudaMalloc(&d_dw_d1,   128 * 128 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_db_d1,   128 * sizeof(float));

        cudaMalloc(&d_dw_d2,   256 * 128 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_db_d2,   256 * sizeof(float));

        cudaMalloc(&d_dw_out,  3 * 256 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_db_out,  3 * sizeof(float));

        CUDA_CHECK(cudaMalloc(&d_loss_accum, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_grad, B * 3 * 32 * 32 * sizeof(float)));

    }


    void init_params() {
        // init host buffers then copy
        int K=3;
        size_t sz_w_e1 = (size_t)ENC1_OUT * IMG_C * K * K;
        size_t sz_w_e2 = (size_t)ENC2_OUT * ENC1_OUT * K * K;
        size_t sz_w_d1 = (size_t)ENC2_OUT * ENC2_OUT * K * K;
        size_t sz_w_d2 = (size_t)ENC1_OUT * ENC2_OUT * K * K;
        size_t sz_w_out = (size_t)IMG_C * ENC1_OUT * K * K;

        std::vector<float> tmp;
        tmp.resize(sz_w_e1);
        for (auto &v: tmp) v = randn();

        // encoder 1
        CUDA_CHECK(cudaMemcpy(d_w_e1, tmp.data(), sz_w_e1*sizeof(float), cudaMemcpyHostToDevice));
        tmp.clear(); tmp.resize(ENC1_OUT);
        for (auto &v: tmp) v = 0.0f;
        CUDA_CHECK(cudaMemcpy(d_b_e1, tmp.data(), ENC1_OUT*sizeof(float), cudaMemcpyHostToDevice));

        // encoder 2
        tmp.clear(); tmp.resize(sz_w_e2);
        for (auto &v: tmp) v = randn();
        CUDA_CHECK(cudaMemcpy(d_w_e2, tmp.data(), sz_w_e2*sizeof(float), cudaMemcpyHostToDevice));
        tmp.clear(); tmp.resize(ENC2_OUT); for (auto &v: tmp) v=0.0f;
        CUDA_CHECK(cudaMemcpy(d_b_e2, tmp.data(), ENC2_OUT*sizeof(float), cudaMemcpyHostToDevice));

        // decoder 1
        tmp.clear(); tmp.resize(sz_w_d1); for (auto &v: tmp) v = randn();
        CUDA_CHECK(cudaMemcpy(d_w_d1, tmp.data(), sz_w_d1*sizeof(float), cudaMemcpyHostToDevice));
        tmp.clear(); tmp.resize(ENC2_OUT); for (auto &v: tmp) v=0.0f;
        CUDA_CHECK(cudaMemcpy(d_b_d1, tmp.data(), ENC2_OUT*sizeof(float), cudaMemcpyHostToDevice));

        // decoder 2
        tmp.clear(); tmp.resize(sz_w_d2); for (auto &v: tmp) v = randn(); 
        CUDA_CHECK(cudaMemcpy(d_w_d2, tmp.data(), sz_w_d2*sizeof(float), cudaMemcpyHostToDevice));
        tmp.clear(); tmp.resize(ENC1_OUT); for (auto &v: tmp) v=0.0f; 
        CUDA_CHECK(cudaMemcpy(d_b_d2, tmp.data(), ENC1_OUT*sizeof(float), cudaMemcpyHostToDevice));

        tmp.clear(); tmp.resize(sz_w_out); for (auto &v: tmp) v = randn();
        CUDA_CHECK(cudaMemcpy(d_w_out, tmp.data(), sz_w_out*sizeof(float), cudaMemcpyHostToDevice));
        tmp.clear(); tmp.resize(IMG_C); for (auto &v: tmp) v=0.0f; 
        CUDA_CHECK(cudaMemcpy(d_b_out, tmp.data(), IMG_C*sizeof(float), cudaMemcpyHostToDevice));

        // zero grads
        int threads = 256;
        int blocks;
        blocks = (sz_w_e1 + threads -1)/threads; zero_kernel<<<blocks,threads>>>(d_dw_e1, sz_w_e1);
        blocks = (ENC1_OUT + threads -1)/threads; zero_kernel<<<blocks,threads>>>(d_db_e1, ENC1_OUT);
        blocks = (sz_w_e2 + threads -1)/threads; zero_kernel<<<blocks,threads>>>(d_dw_e2, sz_w_e2);
        blocks = (ENC2_OUT + threads -1)/threads; zero_kernel<<<blocks,threads>>>(d_db_e2, ENC2_OUT);
        blocks = (sz_w_d1 + threads -1)/threads; zero_kernel<<<blocks,threads>>>(d_dw_d1, sz_w_d1);
        blocks = (ENC2_OUT + threads -1)/threads; zero_kernel<<<blocks,threads>>>(d_db_d1, ENC2_OUT);
        blocks = (sz_w_d2 + threads -1)/threads; zero_kernel<<<blocks,threads>>>(d_dw_d2, sz_w_d2);
        blocks = (ENC1_OUT + threads -1)/threads; zero_kernel<<<blocks,threads>>>(d_db_d2, ENC1_OUT);
        blocks = (sz_w_out + threads -1)/threads; zero_kernel<<<blocks,threads>>>(d_dw_out, sz_w_out);
        blocks = (IMG_C + threads -1)/threads; zero_kernel<<<blocks,threads>>>(d_db_out, IMG_C);

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Forward: input in device memory d_input must be filled already
    void forward() {
        CUDA_CHECK(cudaMemset(d_loss_accum, 0, sizeof(float)));
        int threads = 256;
        // conv1: input (B,3,32,32) -> e1 (B,256,32,32)
        int outH = IMG_H; int outW = IMG_W; int total1 = batch_size * ENC1_OUT * outH * outW;
        int blocks = (total1 + threads -1)/threads;
        conv2d_naive<<<blocks, threads>>>(d_input, batch_size, IMG_C, IMG_H, IMG_W,
                                         d_w_e1, ENC1_OUT, 3, 1, 1, d_e1, outH, outW);
        CUDA_CHECK(cudaPeekAtLastError());
        // add bias
        blocks = (batch_size * ENC1_OUT * outH * outW + threads -1)/threads;
        add_bias<<<blocks, threads>>>(d_e1, d_b_e1, batch_size, ENC1_OUT, outH, outW);
        // relu
        int total_e1 = batch_size * ENC1_OUT * outH * outW;
        blocks = (total_e1 + threads -1)/threads; relu_inplace<<<blocks,threads>>>(d_e1, total_e1);
        // pool1 -> p1 (B,256,16,16)
        int pH = outH / 2; int pW = outW / 2; int total_p1 = batch_size * ENC1_OUT * pH * pW; blocks = (total_p1 + threads -1)/threads;
        maxpool2x2<<<blocks,threads>>>(d_e1, batch_size, ENC1_OUT, outH, outW, d_p1, pH, pW);

        // conv2: p1 (B,256,16,16) -> e2 (B,128,16,16)
        outH = pH; outW = pW; int total_e2 = batch_size * ENC2_OUT * outH * outW; blocks = (total_e2 + threads -1)/threads;
        conv2d_naive<<<blocks,threads>>>(d_p1, batch_size, ENC1_OUT, outH, outW,
                                         d_w_e2, ENC2_OUT, 3, 1, 1, d_e2, outH, outW);
        blocks = (total_e2 + threads -1)/threads; add_bias<<<blocks,threads>>>(d_e2, d_b_e2, batch_size, ENC2_OUT, outH, outW);
        blocks = (total_e2 + threads -1)/threads; relu_inplace<<<blocks,threads>>>(d_e2, total_e2);
        // pool2 -> p2 (B,128,8,8)
        pH = outH/2; pW = outW/2; blocks = (batch_size * ENC2_OUT * pH * pW + threads -1)/threads;
        maxpool2x2<<<blocks,threads>>>(d_e2, batch_size, ENC2_OUT, outH, outW, d_p2, pH, pW);

        // Decoder: conv 128 -> 128 on latent
        int latentH = LAT_H, latentW = LAT_W; int total_dec1 = batch_size * ENC2_OUT * latentH * latentW;
        blocks = (total_dec1 + threads -1)/threads;
        conv2d_naive<<<blocks,threads>>>(d_p2, batch_size, ENC2_OUT, latentH, latentW, d_w_d1, ENC2_OUT, 3, 1, 1, d_dec1, latentH, latentW);
        blocks = (total_dec1 + threads -1)/threads; add_bias<<<blocks,threads>>>(d_dec1, d_b_d1, batch_size, ENC2_OUT, latentH, latentW);
        blocks = (total_dec1 + threads -1)/threads; relu_inplace<<<blocks,threads>>>(d_dec1, total_dec1);
        // upsample -> up1 (B,128,16,16)
        int upH = latentH*2, upW = latentW*2; blocks = (batch_size * ENC2_OUT * upH * upW + threads -1)/threads;
        upsample2x<<<blocks,threads>>>(d_dec1, batch_size, ENC2_OUT, latentH, latentW, d_up1, upH, upW);

        // conv decoder 2: up1 (B,128,16,16) -> dec2 (B,256,16,16)
        int dec2H = upH, dec2W = upW; int total_dec2 = batch_size * ENC1_OUT * dec2H * dec2W; blocks = (total_dec2 + threads -1)/threads;
        conv2d_naive<<<blocks,threads>>>(d_up1, batch_size, ENC2_OUT, dec2H, dec2W, d_w_d2, ENC1_OUT, 3, 1, 1, d_dec2, dec2H, dec2W);
        blocks = (total_dec2 + threads -1)/threads; add_bias<<<blocks,threads>>>(d_dec2, d_b_d2, batch_size, ENC1_OUT, dec2H, dec2W);
        blocks = (total_dec2 + threads -1)/threads; relu_inplace<<<blocks,threads>>>(d_dec2, total_dec2);
        // upsample -> up2 (B,256,32,32)
        int up2H = dec2H*2, up2W = dec2W*2; blocks = (batch_size * ENC1_OUT * up2H * up2W + threads -1)/threads;
        upsample2x<<<blocks,threads>>>(d_dec2, batch_size, ENC1_OUT, dec2H, dec2W, d_up2, up2H, up2W);

        // final conv -> out (B,3,32,32)
        int outHf = up2H, outWf = up2W; int total_out = batch_size * IMG_C * outHf * outWf; blocks = (total_out + threads -1)/threads;
        conv2d_naive<<<blocks,threads>>>(d_up2, batch_size, ENC1_OUT, outHf, outWf, d_w_out, IMG_C, 3, 1, 1, d_out, outHf, outWf);
        blocks = (total_out + threads -1)/threads; add_bias<<<blocks,threads>>>(d_out, d_b_out, batch_size, IMG_C, outHf, outWf);
        // no activation on final

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Backward (naive): compute gradients for final conv only (as example), and update all conv params via SGD with those grads zeroed except final for demonstration.
    // NOTE: Full backward for all layers is long; this implementation computes grads for final conv filters and bias and demonstrates update.
    void backward_and_update_full(float lr, float &h_loss)
    {

        // ------------------------------------------------------
        // 0. Compute dL/dOut from MSE  (out - target)
        // ------------------------------------------------------
        int total_out = batch_size * IMG_C * IMG_H * IMG_W;
        int threads = 256;
        const int block = threads;
        int blocks = (total_out + threads -1)/threads;
        // For demonstration, use target = input (autoencoder)
        mse_loss_and_grad<<<blocks,threads>>>(d_out, d_input, total_out, d_loss_accum, d_out_grad);
        CUDA_CHECK(cudaDeviceSynchronize());

        // copy loss to host
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss_accum, sizeof(float), cudaMemcpyDeviceToHost));
        h_loss /= (float)total_out; // mean
        // printf("MSE loss (batch average): %f\n", h_loss);

        // ------------------------------------------------------
        // BACKPROP
        // ------------------------------------------------------
        // From: conv_out → up2 → dec2 → up1 → dec1 → p2 → e2 → p1 → e1 → input
        // ------------------------------------------------------

        // ======================================================
        // 1. conv_out backward
        // ======================================================

        // dW_out
        conv2d_grad_filters_naive<<< dim3(3, 256), dim3(3, 3) >>>(
            d_up2,          // input
            d_d_out,        // dOut
            batch_size,              // batch
            256, 32, 32,    // input C, H, W
            3, 32, 32,      // output C, H, W
            3, 1, 1,        // kernel 3×3, pad=1, stride=1
            d_dw_out       // dW
        );

        // dB_out
        conv2d_grad_bias<<< 3, 1 >>>(
            d_d_out,
            d_db_out,
            batch_size, 3, 32, 32
        );


        int total_up2 = batch_size * 256 * 16 * 16;
        int grid = (total_up2 + block - 1) / block;
        conv2d_grad_input_naive<<< grid, block >>>(
            d_d_out, d_w_out, d_d_up2,
            batch_size,
            256, 32, 32,
            3, 32, 32,
            3, 1, 1
        );

        // ======================================================
        // 2. up2 backward → grad for dec2
        // ======================================================
        int grid_up2 = (batch_size * 256 * 16 * 16 + block - 1) / block;

        upsample2x_backward<<<grid_up2, block>>>(
            d_d_up2,
            d_d_dec2,
            batch_size, 256, 16, 16
        );

        // ======================================================
        // 3. conv_d2 backward
        // ======================================================

        // dW_d2
        conv2d_grad_filters_naive<<< dim3(256,128), dim3(3,3) >>>(
            d_up1,          // input
            d_d_dec2,       // dOut
            batch_size,
            128, 16, 16,
            256, 16, 16,
            3, 1, 1,
            d_dw_d2
        );

        // dB_d2
        conv2d_grad_bias<<< 256,1 >>>(
            d_d_dec2,
            d_db_d2,
            batch_size,256,16,16
        );

        // d(up1) = d(dec2)
        conv2d_grad_input_naive<<< grid_up2, block >>>(
            d_d_dec2, d_w_d2, d_d_up1,
            batch_size,
            128, 16, 16,
            256, 16, 16,
            3, 1, 1
        );

        // ======================================================
        // 4. up1 backward → grad for dec1
        // ======================================================
        int total_up1 = batch_size * 128 * 8 * 8;
        int grid_up1 = (total_up1 + block - 1) / block;

        upsample2x_backward<<<grid_up1, block>>>(
            d_d_up1,
            d_d_dec1,
            batch_size, 128, 8, 8
        );

        // ======================================================
        // 5. conv_d1 backward
        // ======================================================

        // dW_d1
        conv2d_grad_filters_naive<<< dim3(128,128), dim3(3,3) >>>(
            d_p2,
            d_d_dec1,
            batch_size,
            128,8,8,
            128,8,8,
            3,1,1,
            d_dw_d1
        );

        // dB_d1
        conv2d_grad_bias<<<128,1>>>(
            d_d_dec1,
            d_db_d1,
            batch_size,128,8,8
        );

        // d(p2)
        conv2d_grad_input_naive<<< grid_up1, block >>>(
            d_d_dec1, d_w_d1, d_d_p2,
            batch_size,
            128,8,8,
            128,8,8,
            3,1,1
        );

        // ======================================================
        // 6. maxpool2 backward → grad for e2
        // ======================================================
        int total_p2 = batch_size * 128 * 16 * 16;
        int grid_p2 = (total_p2 + block - 1) / block;

        maxpool2x2_backward<<< grid_p2, block >>>(
            d_e2, d_p2, d_d_p2, d_d_e2,
            batch_size,128,16,16, 8, 8 
        );

        // ======================================================
        // 7. conv_e2 backward
        // ======================================================

        // dW_e2
        conv2d_grad_filters_naive<<< dim3(128,256), dim3(3,3) >>>(
            d_p1,
            d_d_e2,
            batch_size,
            256,16,16,
            128,16,16,
            3,1,1,
            d_dw_e2
        );

        // dB_e2
        conv2d_grad_bias<<<128,1>>>(
            d_d_e2,
            d_db_e2,
            batch_size,128,16,16
        );

        // d(p1)
        conv2d_grad_input_naive<<< grid_p2, block >>>(
            d_d_e2, d_w_e2, d_d_p1,
            batch_size,
            256,16,16,
            128,16,16,
            3,1,1
        );

        // ======================================================
        // 8. maxpool1 backward → grad for e1
        // ======================================================
        int total_p1 = batch_size * 256 * 32 * 32;
        int grid_p1 = (total_p1 + block - 1) / block;

        maxpool2x2_backward<<< grid_p1, block >>>(
            d_e1, d_p1, d_d_p1, d_d_e1,
            batch_size,256,32,32, 16, 16
        );

        // ======================================================
        // 9. conv_e1 backward
        // ======================================================
        // dW_e1
        conv2d_grad_filters_naive<<< dim3(256,3), dim3(3,3) >>>(
            d_input,
            d_d_e1,
            batch_size,
            3,32,32,
            256,32,32,
            3,1,1,
            d_dw_e1
        );

        // dB_e1
        conv2d_grad_bias<<<256,1>>>(
            d_d_e1,
            d_db_e1,
            batch_size,256,32,32
        );

        // d(input)
        conv2d_grad_input_naive<<< grid_p1, block >>>(
            d_d_e1, d_w_e1, d_d_input,
            batch_size,
            3,32,32,
            256,32,32,
            3,1,1
        );

        // ======================================================
        // UPDATE WEIGHTS
        // ======================================================
        update_weights(d_w_e1, d_b_e1, d_dw_e1, d_db_e1, lr, 256*3*3*3, 256);
        update_weights(d_w_e2, d_b_e2, d_dw_e2, d_db_e2, lr, 128*256*3*3, 128);
        update_weights(d_w_d1, d_b_d1, d_dw_d1, d_db_d1, lr, 128*128*3*3, 128);
        update_weights(d_w_d2, d_b_d2, d_dw_d2, d_db_d2, lr, 256*128*3*3, 256);
        update_weights(d_w_out, d_b_out, d_dw_out, d_db_out, lr, 3*256*3*3, 3);
    }
   

    void save_array_to_file(const std::string& filename, float* d_ptr, size_t n) {
        FILE* f = fopen(filename.c_str(), "wb");
        if (!f) { printf("Cannot open %s\n", filename.c_str()); return; }

        std::vector<float> h(n);
        cudaMemcpy(h.data(), d_ptr, n*sizeof(float), cudaMemcpyDeviceToHost);
        fwrite(h.data(), sizeof(float), n, f);
        fclose(f);
    }


    void save_weights_split(const std::string& folder) {
        system(("mkdir -p " + folder).c_str());

        const int K = 3;

        // -------------------------------
        // Layer 1: Conv Layer 1
        // -------------------------------
        save_array_to_file(folder + "/conv1_weight.bin", d_w_e1, ENC1_OUT * IMG_C * K * K);
        save_array_to_file(folder + "/conv1_bias.bin",   d_b_e1, ENC1_OUT);

        // -------------------------------
        // Layer 2: Conv Layer 2
        // -------------------------------
        save_array_to_file(folder + "/conv2_weight.bin", d_w_e2, ENC2_OUT * ENC1_OUT * K * K);
        save_array_to_file(folder + "/conv2_bias.bin",   d_b_e2, ENC2_OUT);

        // -------------------------------
        // Layer 3: Conv Layer 3
        // -------------------------------
        save_array_to_file(folder + "/conv3_weight.bin", d_w_d1, ENC2_OUT * ENC2_OUT * K * K);
        save_array_to_file(folder + "/conv3_bias.bin",   d_b_d1, ENC2_OUT);

        // -------------------------------
        // Layer 4: Conv Layer 4
        // -------------------------------
        save_array_to_file(folder + "/conv4_weight.bin", d_w_d2, ENC1_OUT * ENC2_OUT * K * K);
        save_array_to_file(folder + "/conv4_bias.bin",   d_b_d2, ENC1_OUT);

        // -------------------------------
        // Output Layer: Final Conv
        // -------------------------------
        save_array_to_file(folder + "/conv_out_weight.bin", d_w_out,
                        IMG_C * ENC1_OUT * K * K);
        save_array_to_file(folder + "/conv_out_bias.bin",   d_b_out, IMG_C);

        printf("All weights saved to folder: %s\n", folder.c_str());
    }

    void load_array_from_file(const std::string& filename, float* d_ptr, size_t n) {
        FILE* f = fopen(filename.c_str(), "rb");
        if (!f) {
            printf("Cannot open %s\n", filename.c_str());
            return;
        }

        std::vector<float> h(n);
        fread(h.data(), sizeof(float), n, f);
        fclose(f);

        cudaMemcpy(d_ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    }

    void load_weights_split(const std::string& folder) {
        const int K = 3;

        // -------------------------------
        // Layer 1
        // -------------------------------
        load_array_from_file(folder + "/conv1_weight.bin",
                            d_w_e1, ENC1_OUT * IMG_C * K * K);
        load_array_from_file(folder + "/conv1_bias.bin",
                            d_b_e1, ENC1_OUT);

        // -------------------------------
        // Layer 2
        // -------------------------------
        load_array_from_file(folder + "/conv2_weight.bin",
                            d_w_e2, ENC2_OUT * ENC1_OUT * K * K);
        load_array_from_file(folder + "/conv2_bias.bin",
                            d_b_e2, ENC2_OUT);

        // -------------------------------
        // Layer 3
        // -------------------------------
        load_array_from_file(folder + "/conv3_weight.bin",
                            d_w_d1, ENC2_OUT * ENC2_OUT * K * K);
        load_array_from_file(folder + "/conv3_bias.bin",
                            d_b_d1, ENC2_OUT);

        // -------------------------------
        // Layer 4
        // -------------------------------
        load_array_from_file(folder + "/conv4_weight.bin",
                            d_w_d2, ENC1_OUT * ENC2_OUT * K * K);
        load_array_from_file(folder + "/conv4_bias.bin",
                            d_b_d2, ENC1_OUT);

        // -------------------------------
        // Output layer
        // -------------------------------
        load_array_from_file(folder + "/conv_out_weight.bin",
                            d_w_out, IMG_C * ENC1_OUT * K * K);
        load_array_from_file(folder + "/conv_out_bias.bin",
                            d_b_out, IMG_C);

        printf("All weights loaded from folder: %s\n", folder.c_str());
    }



    ~GPUAutoencoder() {
        cudaFree(d_w_e1); cudaFree(d_b_e1); cudaFree(d_w_e2); cudaFree(d_b_e2);
        cudaFree(d_w_d1); cudaFree(d_b_d1); cudaFree(d_w_d2); cudaFree(d_b_d2);
        cudaFree(d_w_out); cudaFree(d_b_out);
        cudaFree(d_input); cudaFree(d_e1); cudaFree(d_p1); cudaFree(d_e2); cudaFree(d_p2);
        cudaFree(d_dec1); cudaFree(d_up1); cudaFree(d_dec2); cudaFree(d_up2); cudaFree(d_out);
        cudaFree(d_dw_e1); cudaFree(d_db_e1); cudaFree(d_dw_e2); cudaFree(d_db_e2);
        cudaFree(d_dw_d1); cudaFree(d_db_d1); cudaFree(d_dw_d2); cudaFree(d_db_d2);
        cudaFree(d_dw_out); cudaFree(d_db_out);
        cudaFree(d_loss_accum); cudaFree(d_out_grad);
    }
};

// Simple host-side helper to fill input with random images
void fill_random_input(std::vector<float>& h, int batch) {
    std::mt19937 rng(2025);
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    for (auto &v: h) v = d(rng);
}



int main() {
    int batch = 16;  // training batch size
    GPUAutoencoder net(batch);

    // net.load_weights_split("weights");

    // ---- LOAD CIFAR DATA ----
    std::vector<std::vector<float>> train_images;
    std::vector<int> train_labels; // ignored by autoencoder

    if(!load_cifar10_images(
        "../../data/cifar-100-binary/cifar-100-binary/train.bin",
        train_images,
        train_labels,
        50000))
    {
        printf("Load data failed!\n");
        return 0;
    }

    printf("Loaded %zu training images.\n", train_images.size());

    // ---- TRAIN LOOP ----
    int epochs = 5;
    float lr = 1e-5;

    size_t one_img = IMG_C * IMG_H * IMG_W;
    size_t input_sz = (size_t)batch * one_img;

    std::vector<float> h_input(input_sz);


    for (int e = 0; e < epochs; ++e) {
        printf("Epoch %d\n", e);

        float epoch_loss = 0.0f;
        // chạy qua từng batch
        for (size_t i = 0; i + batch <= train_images.size(); i += batch)
        {
            // ----- COPY BATCH → HOST BUFFER -----
            for (int b = 0; b < batch; b++) {
                memcpy(
                    &h_input[b * one_img],
                    train_images[i + b].data(),
                    one_img * sizeof(float)
                );
            }

            // ----- HOST → DEVICE -----
            CUDA_CHECK(cudaMemcpy(
                net.d_input,
                h_input.data(),
                input_sz * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            // ----- FORWARD → BACKWARD → UPDATE -----
            net.forward();
            float h_loss = 0.0;
            net.backward_and_update_full(lr, h_loss);
            epoch_loss +=h_loss;
        }
        size_t num_batches = train_images.size() / batch;
        printf("Epoch %d loss: %f\n", e, epoch_loss / num_batches);

    }

    // ---- SAVE WEIGHTS ----
    net.save_weights_split("weights");

    printf("Done training.\n");
    return 0;
}

