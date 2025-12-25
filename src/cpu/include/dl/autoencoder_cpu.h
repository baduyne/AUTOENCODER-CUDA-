#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include "dl/layers_cpu.h"

namespace dl {

class AutoencoderCPU {
public:
    AutoencoderCPU();
    // Forward: images_in [B, 3, 32, 32], out [B, 3, 32, 32]
    void forward(const float* images_in, float* recon_out, size_t batch_size);
    // Loss giữa 2 ảnh (reconst. loss), trả về giá trị trung bình (MSE)
    float reconstruction_loss(const float* target, const float* recon, size_t batch_size) const;
    // Backward và update weight (performs SGD updates inside conv layers)
    void backward(const float* images, const float* recon, float learning_rate, size_t batch_size);
    // Encoder-only: extract latent representation [B,128,8,8] into out_latent (must point to B*8192 floats)
    void extract_latent(const float* images_in, size_t batch_size, float* out_latent);
    // Latent dimension per sample (C*H*W)
    static constexpr size_t latent_C = 128;
    static constexpr size_t latent_H = 8;
    static constexpr size_t latent_W = 8;
    static constexpr size_t latent_size() { return latent_C * latent_H * latent_W; }
    // Lưu trọng số (delegates to layers)
    void save(const std::string& path) const;
    void load(const std::string& path);

    

private:
    // Encoder layers
    Conv2DCPU conv1_; // 3 -> 256
    ActivationCPU relu1_;
    MaxPool2DCPU pool1_;

    Conv2DCPU conv2_; // 256 -> 128
    ActivationCPU relu2_;
    MaxPool2DCPU pool2_;

    // Decoder layers
    Conv2DCPU dec_conv1_; // 128 -> 128
    ActivationCPU dec_relu1_;
    UpSample2DCPU up1_;

    Conv2DCPU dec_conv2_; // 128 -> 256
    ActivationCPU dec_relu2_;
    UpSample2DCPU up2_;

    Conv2DCPU dec_conv3_; // 256 -> 3 (output)

    // Intermediate buffers (reuse across batches)
    std::vector<float> buf_conv1_;    // [B,256,32,32]
    std::vector<float> buf_act1_;     // [B,256,32,32]
    std::vector<float> buf_pool1_;    // [B,256,16,16]
    std::vector<float> buf_conv2_;    // [B,128,16,16]
    std::vector<float> buf_act2_;     // [B,128,16,16]
    std::vector<float> buf_pool2_;    // [B,128,8,8]  (latent)

    std::vector<float> buf_dec1_;     // [B,128,8,8]
    std::vector<float> buf_dec1_act_; // [B,128,8,8]
    std::vector<float> buf_up1_;      // [B,128,16,16]
    std::vector<float> buf_dec2_;     // [B,256,16,16]
    std::vector<float> buf_dec2_act_; // [B,256,16,16]
    std::vector<float> buf_up2_;      // [B,256,32,32]

    // Gradient buffers
    std::vector<float> grad_buf_conv1_;
    std::vector<float> grad_buf_act1_;
    std::vector<float> grad_buf_pool1_;
    std::vector<float> grad_buf_conv2_;
    std::vector<float> grad_buf_act2_;
    std::vector<float> grad_buf_pool2_;

    std::vector<float> grad_buf_dec1_;
    std::vector<float> grad_buf_dec1_act_;
    std::vector<float> grad_buf_up1_;
    std::vector<float> grad_buf_dec2_;
    std::vector<float> grad_buf_dec2_act_;
    std::vector<float> grad_buf_up2_;
};

} // namespace dl

