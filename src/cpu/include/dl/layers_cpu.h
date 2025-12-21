#pragma once
#include <vector>
#include <string>
#include <cstddef>
#include <memory>

namespace dl {

// (LinearLayerCPU removed — not used by current autoencoder)

// ===== Conv2D Layer =====
class Conv2DCPU {
public:
    Conv2DCPU(int in_channels, int out_channels, int kernel_size, int stride=1, int padding=0);
    void forward(const float* input, float* output, size_t N, int H, int W) const;
    void backward(const float* input, const float* grad_output, float* grad_input,
                  float learning_rate, size_t N, int H, int W);
    void save(const std::string& path) const;
    void load(const std::string& path);
    // Accessors for raw weight/bias buffers (for saving/loading in same layout as GPU)
    const float* weight_data() const { return weight_.data(); }
    float* weight_data() { return weight_.data(); }
    size_t weight_size() const { return weight_.size(); }
    const float* bias_data() const { return bias_.data(); }
    float* bias_data() { return bias_.data(); }
    size_t bias_size() const { return bias_.size(); }
private:
    int Cin_, Cout_, K_, stride_, padding_;
    std::vector<float> weight_; // [Cout, Cin, K, K]
    std::vector<float> bias_;   // [Cout]
};

// ===== MaxPool2D =====
class MaxPool2DCPU {
public:
    MaxPool2DCPU(int kernel, int stride);
    // forward/backward operate with input shape [N, C, H, W]
    void forward(const float* input, float* output, size_t N, int C, int H, int W);
    void backward(const float* grad_output, float* grad_input, size_t N, int C, int H, int W);
private:
    int k_, s_;
    // store last argmax indices (flattened index into input) for backward
    std::vector<int> last_argmax_;
    int last_Hout_ = 0, last_Wout_ = 0;
};

// ===== UpSample2D (nearest) =====
class UpSample2DCPU {
public:
    UpSample2DCPU(int scale);
    void forward(const float* input, float* output, size_t N, int C, int H, int W);
    void backward(const float* grad_output, float* grad_input, size_t N, int C, int H, int W);
private:
    int scale_;
};

// ===== Activation Layer (ReLU, Sigmoid, ...) =====
class ActivationCPU {
public:
    enum Type { ReLU, Sigmoid, None };
    explicit ActivationCPU(Type type);
    void forward(const float* input, float* output, size_t N) const;
    void backward(const float* input, const float* grad_output, float* grad_input, size_t N) const;
private:
    Type type_;
};

} // namespace dl
