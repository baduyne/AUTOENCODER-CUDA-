#include "dl/autoencoder.h"
#include <stdexcept>
#include <iostream>

namespace dl {

Autoencoder::Autoencoder(const torch::Device& device)
    : device_(device),
      // ENCODER
      enc_conv1_(/*in_channels=*/3,
                 /*out_channels=*/256,
                 /*kernel_size=*/3,
                 /*stride=*/1,
                 /*padding=*/1,
                 device),
      enc_conv2_(/*in_channels=*/256,
                 /*out_channels=*/128,
                 /*kernel_size=*/3,
                 /*stride=*/1,
                 /*padding=*/1,
                 device),
      // DECODER
      dec_conv1_(/*in_channels=*/128,
                 /*out_channels=*/128,
                 /*kernel_size=*/3,
                 /*stride=*/1,
                 /*padding=*/1,
                 device),
      dec_conv2_(/*in_channels=*/128,
                 /*out_channels=*/256,
                 /*kernel_size=*/3,
                 /*stride=*/1,
                 /*padding=*/1,
                 device),
      dec_conv_out_(/*in_channels=*/256,
                    /*out_channels=*/3,
                    /*kernel_size=*/3,
                    /*stride=*/1,
                    /*padding=*/1,
                    device)
{
    if (!device_.is_cpu()) {
        throw std::runtime_error(
            "Autoencoder (phase CPU): device must be CPU. "
            "GPU support sẽ thêm ở phase sau."
        );
    }
}

// ===================== ENCODER =====================
// INPUT:  [N, 3, 32, 32]
// OUTPUT: [N, 128, 8, 8]
torch::Tensor Autoencoder::encode(const torch::Tensor& x) {
    if (!x.device().is_cpu()) {
        throw std::runtime_error("Autoencoder::encode: input must be on CPU");
    }
    if (x.dim() != 4 || x.size(1) != 3 || x.size(2) != 32 || x.size(3) != 32) {
        std::cerr << "[WARN] Autoencoder::encode: expected [N,3,32,32], got "
                  << x.sizes() << std::endl;
    }

    x_input_ = x.contiguous();

    // Conv2D(256, 3x3, pad=1, stride=1) + ReLU  -> (N, 256, 32, 32)
    enc_h1_ = enc_conv1_.forward(x_input_);
    enc_h1_ = torch::relu(enc_h1_);

    // MaxPool2D(2x2, stride=2) -> (N, 256, 16, 16)
    enc_p1_ = torch::max_pool2d(enc_h1_, {2, 2}, {2, 2});

    // Conv2D(128, 3x3, pad=1, stride=1) + ReLU -> (N, 128, 16, 16)
    enc_h2_ = enc_conv2_.forward(enc_p1_);
    enc_h2_ = torch::relu(enc_h2_);

    // MaxPool2D(2x2, stride=2) -> (N, 128, 8, 8)
    enc_p2_ = torch::max_pool2d(enc_h2_, {2, 2}, {2, 2});

    latent_ = enc_p2_;

    return latent_;
}

// ===================== DECODER =====================
// INPUT (latent): [N, 128, 8, 8]
// OUTPUT:         [N, 3, 32, 32]
torch::Tensor Autoencoder::decode(const torch::Tensor& z) {
    if (!z.device().is_cpu()) {
        throw std::runtime_error("Autoencoder::decode: latent must be on CPU");
    }
    if (z.dim() != 4 || z.size(1) != 128 || z.size(2) != 8 || z.size(3) != 8) {
        std::cerr << "[WARN] Autoencoder::decode: expected [N,128,8,8], got "
                  << z.sizes() << std::endl;
    }

    latent_ = z.contiguous();

    // Conv2D(128, 3x3, pad=1, stride=1) + ReLU -> (N, 128, 8, 8)
    dec_h1_ = dec_conv1_.forward(latent_);
    dec_h1_ = torch::relu(dec_h1_);

    // UpSample2D(2x2) -> (N, 128, 16, 16)
    dec_up1_ = torch::upsample_nearest2d(dec_h1_, {16, 16});

    // Conv2D(256, 3x3, pad=1, stride=1) + ReLU -> (N, 256, 16, 16)
    dec_h2_ = dec_conv2_.forward(dec_up1_);
    dec_h2_ = torch::relu(dec_h2_);

    // UpSample2D(2x2) -> (N, 256, 32, 32)
    dec_up2_ = torch::upsample_nearest2d(dec_h2_, {32, 32});

    // Conv2D(3, 3x3, pad=1, stride=1) [no activation] -> (N, 3, 32, 32)
    output_ = dec_conv_out_.forward(dec_up2_);

    return output_;
}

// ===================== FORWARD =====================

torch::Tensor Autoencoder::forward(const torch::Tensor& x) {
    auto z = encode(x);
    auto recon = decode(z);
    return recon;
}

// ===================== RECONSTRUCTION LOSS (MSE) =====================

torch::Tensor Autoencoder::reconstruction_loss(const torch::Tensor& input,
                                               const torch::Tensor& reconstruction) {
    return mse_loss_cpu(reconstruction, input);
}

// ===================== EVALUATION (no grad) =====================

double Autoencoder::evaluate_batch(const torch::Tensor& input,
                                   const torch::Tensor& target) {
    if (!input.device().is_cpu() || !target.device().is_cpu()) {
        throw std::runtime_error("Autoencoder::evaluate_batch: tensors must be on CPU");
    }

    torch::NoGradGuard no_grad;
    auto recon = forward(input);
    auto loss = mse_loss_cpu(recon, target);
    return loss.item<double>();
}

// ===================== BACKWARD (using PyTorch autograd) =====================

void Autoencoder::backward(const torch::Tensor& input,
                           const torch::Tensor& target,
                           float learning_rate) {
    if (!input.device().is_cpu() || !target.device().is_cpu()) {
        throw std::runtime_error("Autoencoder::backward: input/target must be on CPU");
    }

    // Enable gradient tracking cho weights
    enc_conv1_.weight().set_requires_grad(true);
    enc_conv1_.bias().set_requires_grad(true);
    enc_conv2_.weight().set_requires_grad(true);
    enc_conv2_.bias().set_requires_grad(true);
    dec_conv1_.weight().set_requires_grad(true);
    dec_conv1_.bias().set_requires_grad(true);
    dec_conv2_.weight().set_requires_grad(true);
    dec_conv2_.bias().set_requires_grad(true);
    dec_conv_out_.weight().set_requires_grad(true);
    dec_conv_out_.bias().set_requires_grad(true);

    // Zero gradients
    if (enc_conv1_.weight().grad().defined()) enc_conv1_.weight().grad().zero_();
    if (enc_conv1_.bias().grad().defined()) enc_conv1_.bias().grad().zero_();
    if (enc_conv2_.weight().grad().defined()) enc_conv2_.weight().grad().zero_();
    if (enc_conv2_.bias().grad().defined()) enc_conv2_.bias().grad().zero_();
    if (dec_conv1_.weight().grad().defined()) dec_conv1_.weight().grad().zero_();
    if (dec_conv1_.bias().grad().defined()) dec_conv1_.bias().grad().zero_();
    if (dec_conv2_.weight().grad().defined()) dec_conv2_.weight().grad().zero_();
    if (dec_conv2_.bias().grad().defined()) dec_conv2_.bias().grad().zero_();
    if (dec_conv_out_.weight().grad().defined()) dec_conv_out_.weight().grad().zero_();
    if (dec_conv_out_.bias().grad().defined()) dec_conv_out_.bias().grad().zero_();

    // Forward pass
    auto recon = forward(input);

    // Compute loss
    auto loss = torch::mse_loss(recon, target);

    // Backward pass - PyTorch autograd
    loss.backward();

    // SGD update với no_grad - kiểm tra gradient tồn tại
    {
        torch::NoGradGuard no_grad;
        
        if (enc_conv1_.weight().grad().defined()) {
            enc_conv1_.weight().sub_(learning_rate * enc_conv1_.weight().grad());
        }
        if (enc_conv1_.bias().grad().defined()) {
            enc_conv1_.bias().sub_(learning_rate * enc_conv1_.bias().grad());
        }
        
        if (enc_conv2_.weight().grad().defined()) {
            enc_conv2_.weight().sub_(learning_rate * enc_conv2_.weight().grad());
        }
        if (enc_conv2_.bias().grad().defined()) {
            enc_conv2_.bias().sub_(learning_rate * enc_conv2_.bias().grad());
        }
        
        if (dec_conv1_.weight().grad().defined()) {
            dec_conv1_.weight().sub_(learning_rate * dec_conv1_.weight().grad());
        }
        if (dec_conv1_.bias().grad().defined()) {
            dec_conv1_.bias().sub_(learning_rate * dec_conv1_.bias().grad());
        }
        
        if (dec_conv2_.weight().grad().defined()) {
            dec_conv2_.weight().sub_(learning_rate * dec_conv2_.weight().grad());
        }
        if (dec_conv2_.bias().grad().defined()) {
            dec_conv2_.bias().sub_(learning_rate * dec_conv2_.bias().grad());
        }
        
        if (dec_conv_out_.weight().grad().defined()) {
            dec_conv_out_.weight().sub_(learning_rate * dec_conv_out_.weight().grad());
        }
        if (dec_conv_out_.bias().grad().defined()) {
            dec_conv_out_.bias().sub_(learning_rate * dec_conv_out_.bias().grad());
        }
    }
}

// ===================== FEATURE EXTRACTION =====================

torch::Tensor Autoencoder::extract_features(const torch::Tensor& x) {
    return encode(x); // latent_ đã được set bên trong encode()
}

// ===================== SAVE / LOAD WEIGHTS =====================

void Autoencoder::save(const std::string& path) const {
    // Lưu tất cả weight + bias của 5 conv layers theo thứ tự cố định
    std::vector<torch::Tensor> params = {
        enc_conv1_.weight(), enc_conv1_.bias(),
        enc_conv2_.weight(), enc_conv2_.bias(),
        dec_conv1_.weight(), dec_conv1_.bias(),
        dec_conv2_.weight(), dec_conv2_.bias(),
        dec_conv_out_.weight(), dec_conv_out_.bias()
    };

    torch::save(params, path);
}

void Autoencoder::load(const std::string& path) {
    std::vector<torch::Tensor> params;
    torch::load(params, path);

    if (params.size() != 10) {
        throw std::runtime_error("Autoencoder::load: expected 10 tensors in checkpoint");
    }

    // Copy vào weight/bias tương ứng
    // Perform copies under NoGradGuard to avoid in-place ops on leaf tensors
    // that require grad (autograd disallows such in-place modifications).
    {
        torch::NoGradGuard no_grad;

        enc_conv1_.weight().copy_(params[0]);
        enc_conv1_.bias().copy_(params[1]);

        enc_conv2_.weight().copy_(params[2]);
        enc_conv2_.bias().copy_(params[3]);

        dec_conv1_.weight().copy_(params[4]);
        dec_conv1_.bias().copy_(params[5]);

        dec_conv2_.weight().copy_(params[6]);
        dec_conv2_.bias().copy_(params[7]);

        dec_conv_out_.weight().copy_(params[8]);
        dec_conv_out_.bias().copy_(params[9]);
    }
}

// ===================== TO (device) =====================

void Autoencoder::to(const torch::Device& device) {
    if (!device.is_cpu()) {
        throw std::runtime_error(
            "Autoencoder::to: phase hiện tại chỉ hỗ trợ CPU. "
            "GPU sẽ implement ở phase sau."
        );
    }
    device_ = device;
}

} // namespace dl
