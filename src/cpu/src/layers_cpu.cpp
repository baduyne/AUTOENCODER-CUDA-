#include "dl/layers_cpu.h"
#include <cmath>
#include <fstream>
#include <cassert>
#include <cstring>
#include <random>
#include <algorithm>
#include <iostream>
#include "dl/conv2d_cpu.h"

namespace dl {

// (LinearLayerCPU implementation removed; autoencoder uses Conv/Pool/UpSample/Activation only)

// ===== ActivationCPU =====
ActivationCPU::ActivationCPU(Type type) : type_(type) {}
void ActivationCPU::forward(const float* input, float* output, size_t N) const {
    switch(type_) {
    case ReLU:
        for(size_t i=0;i<N;++i) output[i]=input[i]>0?input[i]:0;
        break;
    case Sigmoid:
        for(size_t i=0;i<N;++i) output[i]=1.f/(1.f+std::exp(-input[i]));
        break;
    case None:
    default:
        memcpy(output, input, N*sizeof(float));
    }
}
void ActivationCPU::backward(const float* input, const float* grad_output, float* grad_input, size_t N) const {
    switch(type_) {
    case ReLU:
        for(size_t i=0;i<N;++i) grad_input[i]=input[i]>0?grad_output[i]:0.f;
        break;
    case Sigmoid:
        for(size_t i=0;i<N;++i) {
            float sig=1.f/(1.f+std::exp(-input[i]));
            grad_input[i]=grad_output[i]*sig*(1.f-sig);
        }
        break;
    case None:
    default:
        memcpy(grad_input, grad_output, N*sizeof(float));
    }
}

} // namespace dl

// ===== Conv2DCPU implementation =====
namespace dl {

Conv2DCPU::Conv2DCPU(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : Cin_(in_channels), Cout_(out_channels), K_(kernel_size), stride_(stride), padding_(padding),
      weight_(out_channels * in_channels * kernel_size * kernel_size), bias_(out_channels)
{
    // Xavier (Glorot) uniform init for conv weights, bias = 0
    std::mt19937 gen(42);
    int kernel_area = K_ * K_;
    float denom = static_cast<float>(in_channels * kernel_area + out_channels * kernel_area);
    float limit = std::sqrt(6.0f / std::max(1.0f, denom));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto &w : weight_) w = dist(gen);
    for (auto &b : bias_) b = 0.0f;
}

void Conv2DCPU::forward(const float* input, float* output, size_t N, int H, int W) const {
    conv2d_forward_cpu(input, weight_.data(), bias_.data(), output,
                       static_cast<int>(N), Cin_, H, W, Cout_, K_, stride_, padding_);
}

void Conv2DCPU::backward(const float* input, const float* grad_output, float* grad_input,
                         float learning_rate, size_t N, int H, int W) {
    std::vector<float> grad_weight(weight_.size());
    std::vector<float> grad_bias(Cout_);
    conv2d_backward_cpu(input, weight_.data(), grad_output, grad_input,
                        grad_weight.data(), grad_bias.data(),
                        static_cast<int>(N), Cin_, H, W, Cout_, K_, stride_, padding_);
        const char* dbg_env = std::getenv("CHECK_GRAD");
        bool do_dbg = dbg_env && dbg_env[0];
        double w_norm_before = 0.0, b_norm_before = 0.0;
        double gw_norm = 0.0, gb_norm = 0.0;
        if (do_dbg) {
            for (size_t i = 0; i < weight_.size(); ++i) w_norm_before += static_cast<double>(weight_[i]) * weight_[i];
            for (size_t i = 0; i < bias_.size(); ++i) b_norm_before += static_cast<double>(bias_[i]) * bias_[i];
            for (size_t i = 0; i < grad_weight.size(); ++i) gw_norm += static_cast<double>(grad_weight[i]) * grad_weight[i];
            for (size_t i = 0; i < grad_bias.size(); ++i) gb_norm += static_cast<double>(grad_bias[i]) * grad_bias[i];
            w_norm_before = std::sqrt(w_norm_before);
            b_norm_before = std::sqrt(b_norm_before);
            gw_norm = std::sqrt(gw_norm);
            gb_norm = std::sqrt(gb_norm);
            std::cerr << "[DEBUG] Conv2DCPU::backward PRE update grads_norm=" << gw_norm << " bias_grads_norm=" << gb_norm
                      << " weights_norm=" << w_norm_before << " bias_norm=" << b_norm_before << "\n";
        }

        // Apply update using summed gradients (match GPU semantics)
        for (size_t i = 0; i < bias_.size(); ++i) bias_[i] -= static_cast<float>(learning_rate * grad_bias[i]);
        for (size_t i = 0; i < weight_.size(); ++i) weight_[i] -= static_cast<float>(learning_rate * grad_weight[i]);

        if (do_dbg) {
            double w_norm_after = 0.0, b_norm_after = 0.0;
            for (size_t i = 0; i < weight_.size(); ++i) w_norm_after += static_cast<double>(weight_[i]) * weight_[i];
            for (size_t i = 0; i < bias_.size(); ++i) b_norm_after += static_cast<double>(bias_[i]) * bias_[i];
            w_norm_after = std::sqrt(w_norm_after);
            b_norm_after = std::sqrt(b_norm_after);
            std::cerr << "[DEBUG] Conv2DCPU::backward POST update weights_norm=" << w_norm_after << " bias_norm=" << b_norm_after
                      << " (delta_w=" << (w_norm_after - w_norm_before) << ")\n";
        }
}

void Conv2DCPU::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    f.write(reinterpret_cast<const char*>(&Cin_), sizeof(Cin_));
    f.write(reinterpret_cast<const char*>(&Cout_), sizeof(Cout_));
    f.write(reinterpret_cast<const char*>(&K_), sizeof(K_));
    f.write(reinterpret_cast<const char*>(weight_.data()), weight_.size()*sizeof(float));
    f.write(reinterpret_cast<const char*>(bias_.data()), bias_.size()*sizeof(float));
}

void Conv2DCPU::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return;
    int cin=0, cout=0, k=0;
    f.read(reinterpret_cast<char*>(&cin), sizeof(cin));
    f.read(reinterpret_cast<char*>(&cout), sizeof(cout));
    f.read(reinterpret_cast<char*>(&k), sizeof(k));
    Cin_ = cin; Cout_ = cout; K_ = k;
    weight_.resize(static_cast<size_t>(Cout_) * Cin_ * K_ * K_);
    bias_.resize(static_cast<size_t>(Cout_));
    f.read(reinterpret_cast<char*>(weight_.data()), weight_.size()*sizeof(float));
    f.read(reinterpret_cast<char*>(bias_.data()), bias_.size()*sizeof(float));
}

// ===== MaxPool2DCPU implementation =====
MaxPool2DCPU::MaxPool2DCPU(int kernel, int stride) : k_(kernel), s_(stride) {}

void MaxPool2DCPU::forward(const float* input, float* output, size_t N, int C, int H, int W) {
    int Hout = (H - k_) / s_ + 1;
    int Wout = (W - k_) / s_ + 1;
    last_Hout_ = Hout; last_Wout_ = Wout;
    last_argmax_.assign(static_cast<size_t>(N) * C * Hout * Wout, -1);
    for (int n = 0; n < static_cast<int>(N); ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < Hout; ++oh) {
                for (int ow = 0; ow < Wout; ++ow) {
                    float best = -std::numeric_limits<float>::infinity();
                    int best_idx = -1;
                    int ih0 = oh * s_;
                    int iw0 = ow * s_;
                    for (int kh = 0; kh < k_; ++kh) {
                        for (int kw = 0; kw < k_; ++kw) {
                            int ih = ih0 + kh;
                            int iw = iw0 + kw;
                            int in_idx = ((n * C + c) * H + ih) * W + iw;
                            float v = input[in_idx];
                            if (v > best) { best = v; best_idx = in_idx; }
                        }
                    }
                    int out_idx = ((n * C + c) * Hout + oh) * Wout + ow;
                    output[out_idx] = best;
                    last_argmax_[out_idx] = best_idx;
                }
            }
        }
    }
}

void MaxPool2DCPU::backward(const float* grad_output, float* grad_input, size_t N, int C, int H, int W) {
    std::fill(grad_input, grad_input + static_cast<size_t>(N) * C * H * W, 0.0f);
    int Hout = last_Hout_, Wout = last_Wout_;
    size_t out_size = static_cast<size_t>(N) * C * Hout * Wout;
    for (size_t idx = 0; idx < out_size; ++idx) {
        int in_idx = last_argmax_[idx];
        if (in_idx >= 0) grad_input[static_cast<size_t>(in_idx)] += grad_output[idx];
    }
}

// ===== UpSample2DCPU implementation (nearest) =====
UpSample2DCPU::UpSample2DCPU(int scale) : scale_(scale) {}

void UpSample2DCPU::forward(const float* input, float* output, size_t N, int C, int H, int W) {
    int Hout = H * scale_;
    int Wout = W * scale_;
    for (int n = 0; n < static_cast<int>(N); ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < Hout; ++oh) {
                int ih = oh / scale_;
                for (int ow = 0; ow < Wout; ++ow) {
                    int iw = ow / scale_;
                    int in_idx = ((n * C + c) * H + ih) * W + iw;
                    int out_idx = ((n * C + c) * Hout + oh) * Wout + ow;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

void UpSample2DCPU::backward(const float* grad_output, float* grad_input, size_t N, int C, int H, int W) {
    int Hout = H * scale_;
    int Wout = W * scale_;
    std::fill(grad_input, grad_input + static_cast<size_t>(N) * C * H * W, 0.0f);
    for (int n = 0; n < static_cast<int>(N); ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < Hout; ++oh) {
                int ih = oh / scale_;
                for (int ow = 0; ow < Wout; ++ow) {
                    int iw = ow / scale_;
                    int in_idx = ((n * C + c) * H + ih) * W + iw;
                    int out_idx = ((n * C + c) * Hout + oh) * Wout + ow;
                    grad_input[in_idx] += grad_output[out_idx];
                }
            }
        }
    }
}

} // namespace dl
