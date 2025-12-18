    #pragma once
    #include <torch/torch.h>
    #include <string>
    #include "dl/device.h"
    #include "dl/layers_cpu.h"

    namespace dl {

    // Autoencoder CPU
    // INPUT:  [N, 3, 32, 32]
    // LATENT: [N, 128, 8, 8]
    // OUTPUT: [N, 3, 32, 32]
    class Autoencoder {
    public:
        explicit Autoencoder(const torch::Device& device);

        // ----- Forward path -----
        // Encode: (N, 3, 32, 32) -> (N, 128, 8, 8)
        torch::Tensor encode(const torch::Tensor& x);

        // Decode: (N, 128, 8, 8) -> (N, 3, 32, 32)
        torch::Tensor decode(const torch::Tensor& z);

        // Full forward: input -> reconstruction
        torch::Tensor forward(const torch::Tensor& x);

        // Reconstruction loss (MSE) giữa input và reconstruction
        torch::Tensor reconstruction_loss(const torch::Tensor& input,
                                        const torch::Tensor& reconstruction);

        // ----- Backward path (skeleton) -----
        // Tính loss và chuẩn bị gradient top-level (dL/dOutput).
        // Phase hiện tại CHƯA có backward cho từng layer, nên đây là khung để sau này
        // bạn gọi các hàm backward tương ứng (conv, pool, upsample, relu).
        void backward(const torch::Tensor& input,
                    const torch::Tensor& target,
                    float learning_rate);

        // ----- Feature extraction -----
        // Chạy encoder, trả về latent representation [N, 128, 8, 8]
        torch::Tensor extract_features(const torch::Tensor& x);

        // Trả về latent của lần forward/encode gần nhất
        const torch::Tensor& latent() const { return latent_; }

        // ----- Save / Load weights -----
        // Lưu tất cả weight + bias của 5 conv layer ra file
        void save(const std::string& path) const;

        // Load trọng số từ file
        void load(const std::string& path);

        // ----- Device -----
        void to(const torch::Device& device);
        torch::Device device() const { return device_; }

        // ----- Evaluation -----
        // Tính reconstruction loss (MSE) trên một batch (không cập nhật weights).
        // Trả về giá trị loss dưới dạng double (scalar)
        double evaluate_batch(const torch::Tensor& input,
                            const torch::Tensor& target);

    private:
        torch::Device device_;

        // ENCODER
        // Conv1: 3 -> 256, 3x3, pad=1, stride=1
        // Conv2: 256 -> 128, 3x3, pad=1, stride=1
        Conv2D enc_conv1_; // 3   -> 256
        Conv2D enc_conv2_; // 256 -> 128

        // DECODER
        // DecConv1: 128 -> 128
        // DecConv2: 128 -> 256
        // DecOut:   256 -> 3
        Conv2D dec_conv1_;    // 128 -> 128
        Conv2D dec_conv2_;    // 128 -> 256
        Conv2D dec_conv_out_; // 256 -> 3

        // ----- Intermediate activations (lưu forward) -----
        // Encoder
        torch::Tensor x_input_;  // [N, 3, 32, 32]
        torch::Tensor enc_h1_;   // after enc_conv1 + ReLU: [N, 256, 32, 32]
        torch::Tensor enc_p1_;   // after pool1:           [N, 256, 16, 16]
        torch::Tensor enc_h2_;   // after enc_conv2 + ReLU:[N, 128, 16, 16]
        torch::Tensor enc_p2_;   // after pool2 (latent):  [N, 128, 8, 8]

        // Latent
        torch::Tensor latent_;   // alias enc_p2_

        // Decoder
        torch::Tensor dec_h1_;   // after dec_conv1 + ReLU: [N, 128, 8, 8]
        torch::Tensor dec_up1_;  // after upsample:         [N, 128, 16, 16]
        torch::Tensor dec_h2_;   // after dec_conv2 + ReLU: [N, 256, 16, 16]
        torch::Tensor dec_up2_;  // after upsample:         [N, 256, 32, 32]
        torch::Tensor output_;   // after dec_conv_out:     [N, 3, 32, 32]
    };

    } // namespace dl
