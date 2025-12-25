#include "dl/autoencoder_cpu.h"
#include <fstream>
#include <iostream>
#include <cassert>
#include <cstring>

namespace dl {

AutoencoderCPU::AutoencoderCPU()
    : conv1_(/*in*/3, /*out*/256, /*k*/3, /*s*/1, /*p*/1), relu1_(ActivationCPU::ReLU), pool1_(2,2),
      conv2_(256, 128, 3, 1, 1), relu2_(ActivationCPU::ReLU), pool2_(2,2),
      dec_conv1_(128, 128, 3, 1, 1), dec_relu1_(ActivationCPU::ReLU), up1_(2),
      dec_conv2_(128, 256, 3, 1, 1), dec_relu2_(ActivationCPU::ReLU), up2_(2),
      dec_conv3_(256, 3, 3, 1, 1)
{
}

// Helper to ensure buffer sizes
static void ensure_size(std::vector<float>& v, size_t sz) { if (v.size() < sz) v.resize(sz); }

void AutoencoderCPU::forward(const float* images_in, float* recon_out, size_t batch_size) {
    const int N = static_cast<int>(batch_size);
    static bool dbg_once = false;
    if (dbg_once) {
        // std::cerr << "[DEBUG] AutoencoderCPU::forward called (batch_size=" << batch_size << ")\n";
        dbg_once = false;
    }
    // shapes
    const int H = 32, W = 32;
    const int C1 = 256;
    const int C2 = 128;
    // allocate buffers
    ensure_size(buf_conv1_, static_cast<size_t>(N) * C1 * H * W);
    ensure_size(buf_act1_, static_cast<size_t>(N) * C1 * H * W);
    ensure_size(buf_pool1_, static_cast<size_t>(N) * C1 * (H/2) * (W/2));
    ensure_size(buf_conv2_, static_cast<size_t>(N) * C2 * (H/2) * (W/2));
    ensure_size(buf_act2_, static_cast<size_t>(N) * C2 * (H/2) * (W/2));
    ensure_size(buf_pool2_, static_cast<size_t>(N) * C2 * (H/4) * (W/4)); // latent

    ensure_size(buf_dec1_, static_cast<size_t>(N) * C2 * (H/4) * (W/4));
    ensure_size(buf_dec1_act_, static_cast<size_t>(N) * C2 * (H/4) * (W/4));
    ensure_size(buf_up1_, static_cast<size_t>(N) * C2 * (H/2) * (W/2));
    ensure_size(buf_dec2_, static_cast<size_t>(N) * C1 * (H/2) * (W/2));
    ensure_size(buf_dec2_act_, static_cast<size_t>(N) * C1 * (H/2) * (W/2));
    ensure_size(buf_up2_, static_cast<size_t>(N) * C1 * H * W);

    // Encoder
    conv1_.forward(images_in, buf_conv1_.data(), batch_size, H, W);
    relu1_.forward(buf_conv1_.data(), buf_act1_.data(), buf_conv1_.size());
    pool1_.forward(buf_act1_.data(), buf_pool1_.data(), batch_size, C1, H, W);

    conv2_.forward(buf_pool1_.data(), buf_conv2_.data(), batch_size, H/2, W/2);
    relu2_.forward(buf_conv2_.data(), buf_act2_.data(), buf_conv2_.size());
    pool2_.forward(buf_act2_.data(), buf_pool2_.data(), batch_size, C2, H/2, W/2);

    // Decoder
    dec_conv1_.forward(buf_pool2_.data(), buf_dec1_.data(), batch_size, H/4, W/4);
    dec_relu1_.forward(buf_dec1_.data(), buf_dec1_act_.data(), buf_dec1_.size());
    up1_.forward(buf_dec1_act_.data(), buf_up1_.data(), batch_size, C2, H/4, W/4);

    dec_conv2_.forward(buf_up1_.data(), buf_dec2_.data(), batch_size, H/2, W/2);
    dec_relu2_.forward(buf_dec2_.data(), buf_dec2_act_.data(), buf_dec2_.size());
    up2_.forward(buf_dec2_act_.data(), buf_up2_.data(), batch_size, C1, H/2, W/2);

    dec_conv3_.forward(buf_up2_.data(), recon_out, batch_size, H, W);
    // if constexpr (true) std::cerr << "[DEBUG] AutoencoderCPU::forward completed for batch=" << batch_size << "\n";
}

void AutoencoderCPU::extract_latent(const float* images_in, size_t batch_size, float* out_latent) {
    // Run encoder only, fill out_latent with shape [B, 128, 8, 8] (NCHW flattened)
    const int N = static_cast<int>(batch_size);
    const int H = 32, W = 32;
    const int C1 = 256, C2 = 128;
    ensure_size(buf_conv1_, static_cast<size_t>(N) * C1 * H * W);
    ensure_size(buf_act1_, static_cast<size_t>(N) * C1 * H * W);
    ensure_size(buf_pool1_, static_cast<size_t>(N) * C1 * (H/2) * (W/2));
    ensure_size(buf_conv2_, static_cast<size_t>(N) * C2 * (H/2) * (W/2));
    ensure_size(buf_act2_, static_cast<size_t>(N) * C2 * (H/2) * (W/2));
    ensure_size(buf_pool2_, static_cast<size_t>(N) * C2 * (H/4) * (W/4)); // latent

    conv1_.forward(images_in, buf_conv1_.data(), batch_size, H, W);
    relu1_.forward(buf_conv1_.data(), buf_act1_.data(), buf_conv1_.size());
    pool1_.forward(buf_act1_.data(), buf_pool1_.data(), batch_size, C1, H, W);

    conv2_.forward(buf_pool1_.data(), buf_conv2_.data(), batch_size, H/2, W/2);
    relu2_.forward(buf_conv2_.data(), buf_act2_.data(), buf_conv2_.size());
    pool2_.forward(buf_act2_.data(), buf_pool2_.data(), batch_size, C2, H/2, W/2);

    // copy latent to out_latent
    size_t latent_elems = static_cast<size_t>(batch_size) * C2 * (H/4) * (W/4);
    std::memcpy(out_latent, buf_pool2_.data(), latent_elems * sizeof(float));
}

// (duplicate extract_latent removed)

float AutoencoderCPU::reconstruction_loss(const float* target, const float* recon, size_t batch_size) const {
    size_t total = batch_size * 3 * 32 * 32;
    double loss = 0.0;
    for (size_t i = 0; i < total; ++i) {
        double diff = static_cast<double>(target[i]) - static_cast<double>(recon[i]);
        loss += diff * diff;
    }
    return static_cast<float>(loss / static_cast<double>(total));
}

void AutoencoderCPU::backward(const float* images, const float* recon, float learning_rate, size_t batch_size) {
    const int N = static_cast<int>(batch_size);
    static bool dbg_once_b = true;
    if (dbg_once_b) {
        // std::cerr << "[DEBUG] AutoencoderCPU::backward called (batch_size=" << batch_size << ")\n";
        dbg_once_b = false;
    }
    const int H = 32, W = 32;
    const int C1 = 256;
    const int C2 = 128;
    // allocate gradient buffers
    ensure_size(grad_buf_up2_, static_cast<size_t>(N) * C1 * H * W);
    ensure_size(grad_buf_dec2_, static_cast<size_t>(N) * C1 * (H/2) * (W/2));
    ensure_size(grad_buf_up1_, static_cast<size_t>(N) * C2 * (H/2) * (W/2));
    ensure_size(grad_buf_dec1_, static_cast<size_t>(N) * C2 * (H/4) * (W/4));
    ensure_size(grad_buf_pool2_, static_cast<size_t>(N) * C2 * (H/4) * (W/4));
    ensure_size(grad_buf_act2_, static_cast<size_t>(N) * C2 * (H/2) * (W/2));
    ensure_size(grad_buf_conv2_, static_cast<size_t>(N) * C2 * (H/2) * (W/2));
    ensure_size(grad_buf_pool1_, static_cast<size_t>(N) * C1 * (H/2) * (W/2));
    ensure_size(grad_buf_act1_, static_cast<size_t>(N) * C1 * H * W);
    ensure_size(grad_buf_conv1_, static_cast<size_t>(N) * C1 * H * W);
    // gradient wrt original input images (unused but required by conv.backward)
    std::vector<float> grad_buf_input_images(static_cast<size_t>(N) * 3 * H * W);

    // dL/drecon = 2*(recon - target)/total
    size_t total = static_cast<size_t>(batch_size) * 3 * H * W;
    std::vector<float> grad_recon(total);
    for (size_t i = 0; i < total; ++i) grad_recon[i] = 2.0f * (recon[i] - images[i]) / static_cast<float>(total);

    // Decoder backward
    // dec_conv3: input buf_up2_ -> output recon
    dec_conv3_.backward(buf_up2_.data(), grad_recon.data(), grad_buf_up2_.data(), learning_rate, batch_size, H, W);

    // up2 backward: grad_buf_up2_ -> grad_buf_dec2_
    up2_.backward(grad_buf_up2_.data(), grad_buf_dec2_.data(), batch_size, C1, H/2, W/2);

    // dec_conv2 backward: input buf_up1_ -> output buf_dec2_
    dec_conv2_.backward(buf_up1_.data(), grad_buf_dec2_.data(), grad_buf_up1_.data(), learning_rate, batch_size, H/2, W/2);

    // dec_relu2 backward: input buf_dec2_ -> grad_buf_up1_
    dec_relu2_.backward(buf_dec2_.data(), grad_buf_up1_.data(), grad_buf_dec2_.data(), buf_dec2_.size());

    // up1 backward -> grad_buf_dec1_
    up1_.backward(grad_buf_dec2_.data(), grad_buf_dec1_.data(), batch_size, C2, H/4, W/4);

    // dec_conv1 backward: input buf_pool2_ -> output buf_dec1_
    dec_conv1_.backward(buf_pool2_.data(), grad_buf_dec1_.data(), grad_buf_pool2_.data(), learning_rate, batch_size, H/4, W/4);

    // dec_relu1 backward -> grad_buf_pool2_
    dec_relu1_.backward(buf_dec1_.data(), grad_buf_pool2_.data(), grad_buf_dec1_.data(), buf_dec1_.size());

    // Encoder backward
    pool2_.backward(grad_buf_pool2_.data(), grad_buf_act2_.data(), batch_size, C2, H/2, W/2);

    // relu2 backward -> grad_buf_conv2_
    relu2_.backward(buf_conv2_.data(), grad_buf_act2_.data(), grad_buf_conv2_.data(), buf_conv2_.size());

    // conv2 backward: input buf_pool1_ -> output grad_buf_pool1_
    conv2_.backward(buf_pool1_.data(), grad_buf_conv2_.data(), grad_buf_pool1_.data(), learning_rate, batch_size, H/2, W/2);

    // pool1 backward -> grad_buf_act1_
    pool1_.backward(grad_buf_pool1_.data(), grad_buf_act1_.data(), batch_size, C1, H, W);

    // relu1 backward -> grad_buf_conv1_
    relu1_.backward(buf_conv1_.data(), grad_buf_act1_.data(), grad_buf_conv1_.data(), buf_conv1_.size());

    // conv1 backward: input images -> grad to input (stored in grad_buf_input_images but ignored)
    conv1_.backward(images, grad_buf_conv1_.data(), grad_buf_input_images.data(), learning_rate, batch_size, H, W);
    // std::cerr << "[DEBUG] AutoencoderCPU::backward completed for batch=" << batch_size << "\n";
}

void AutoencoderCPU::save(const std::string& path) const {
    // Save all conv weights/biases into a single binary file (same layout as GPU save_weights)
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { std::cerr << "[ERR] failed to open " << path << " for writing\n"; return; }

    // Order: enc_conv1_w, enc_conv1_b, enc_conv2_w, enc_conv2_b,
    //        dec_conv1_w, dec_conv1_b, dec_conv2_w, dec_conv2_b,
    //        dec_conv3_w, dec_conv3_b
    fwrite(const_cast<float*>(conv1_.weight_data()), sizeof(float), conv1_.weight_size(), f);
    fwrite(const_cast<float*>(conv1_.bias_data()),   sizeof(float), conv1_.bias_size(),   f);
    fwrite(const_cast<float*>(conv2_.weight_data()), sizeof(float), conv2_.weight_size(), f);
    fwrite(const_cast<float*>(conv2_.bias_data()),   sizeof(float), conv2_.bias_size(),   f);
    fwrite(const_cast<float*>(dec_conv1_.weight_data()), sizeof(float), dec_conv1_.weight_size(), f);
    fwrite(const_cast<float*>(dec_conv1_.bias_data()),   sizeof(float), dec_conv1_.bias_size(),   f);
    fwrite(const_cast<float*>(dec_conv2_.weight_data()), sizeof(float), dec_conv2_.weight_size(), f);
    fwrite(const_cast<float*>(dec_conv2_.bias_data()),   sizeof(float), dec_conv2_.bias_size(),   f);
    fwrite(const_cast<float*>(dec_conv3_.weight_data()), sizeof(float), dec_conv3_.weight_size(), f);
    fwrite(const_cast<float*>(dec_conv3_.bias_data()),   sizeof(float), dec_conv3_.bias_size(),   f);

    fclose(f);
}

void AutoencoderCPU::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { std::cerr << "[ERR] failed to open " << path << " for reading\n"; return; }

    // Ensure host storage sizes match expected; resize vectors if necessary
    // Read in same order as save
    size_t n;

    n = conv1_.weight_size(); fread(conv1_.weight_data(), sizeof(float), n, f);
    n = conv1_.bias_size();   fread(conv1_.bias_data(),   sizeof(float), n, f);
    n = conv2_.weight_size(); fread(conv2_.weight_data(), sizeof(float), n, f);
    n = conv2_.bias_size();   fread(conv2_.bias_data(),   sizeof(float), n, f);
    n = dec_conv1_.weight_size(); fread(dec_conv1_.weight_data(), sizeof(float), n, f);
    n = dec_conv1_.bias_size();   fread(dec_conv1_.bias_data(),   sizeof(float), n, f);
    n = dec_conv2_.weight_size(); fread(dec_conv2_.weight_data(), sizeof(float), n, f);
    n = dec_conv2_.bias_size();   fread(dec_conv2_.bias_data(),   sizeof(float), n, f);
    n = dec_conv3_.weight_size(); fread(dec_conv3_.weight_data(), sizeof(float), n, f);
    n = dec_conv3_.bias_size();   fread(dec_conv3_.bias_data(),   sizeof(float), n, f);

    fclose(f);

    // No device copy here; training pipeline uses CPU layers. If GPU parity is needed,
    // one can copy these buffers to GPU via existing GPU class routines.
}

} // namespace dl

