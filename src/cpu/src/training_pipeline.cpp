#include "dl/training_pipeline.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <algorithm> // std::max
 #include <tuple>

namespace dl {

using Clock = std::chrono::high_resolution_clock;

TrainingPipeline::TrainingPipeline(const std::string& cifar_root,
                                   torch::Device device)
    : cifar_root_(cifar_root),
      device_(device),
      autoencoder_(device) {}

// ===== Phase 1: Train Autoencoder =====
void TrainingPipeline::train_autoencoder(int   num_epochs,
                                         int   batch_size,
                                         float learning_rate,
                                         const std::string& ae_save_path)
{
    std::cout << "===== TRAINING AUTOENCODER (CPU) =====" << std::endl;

    CIFAR10Dataset train_set(cifar_root_, /*train=*/true);
    std::cout << "Train size: " << train_set.size() << " samples" << std::endl;

    torch::Tensor images, labels;

    // Tính tổng số batch cho progress tracking
    const int total_samples = train_set.size();
    const int batches_per_epoch = (total_samples + batch_size - 1) / batch_size;

    std::cout << "Total samples: " << total_samples 
              << ", Batches per epoch: " << batches_per_epoch 
              << ", Batch size: " << batch_size << std::endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = Clock::now();

        train_set.reset(/*shuffle=*/true);

        // Validation set used for quick eval checks during epoch
        CIFAR10Dataset val_set(cifar_root_, /*train=*/false);
        val_set.reset(/*shuffle=*/false);

        double epoch_loss_sum = 0.0;
        int    num_batches    = 0;

        // Aggregation for periodic logging (every N batches)
        const int log_interval = 50;
        const int eval_interval = 50;
        double agg_loss = 0.0;
        double agg_fwd_time = 0.0;
        double agg_bwd_time = 0.0;
        double agg_batch_time = 0.0;
        int    agg_count = 0;

        while (train_set.next_batch(batch_size, device_, images, labels)) {
            auto batch_start = Clock::now();

            auto forward_start = Clock::now();
            // Forward
            auto recon = autoencoder_.forward(images);
            auto loss  = autoencoder_.reconstruction_loss(images, recon);
            auto forward_end = Clock::now();

            double batch_loss = loss.item<double>();
            epoch_loss_sum += batch_loss;
            ++num_batches;

            auto backward_start = Clock::now();
            // Backward + update
            autoencoder_.backward(images, images, learning_rate);
            auto backward_end = Clock::now();

            auto batch_end = Clock::now();

            std::chrono::duration<double> forward_time = forward_end - forward_start;
            std::chrono::duration<double> backward_time = backward_end - backward_start;
            std::chrono::duration<double> batch_time = batch_end - batch_start;

            // Aggregate for periodic logging
            agg_loss += batch_loss;
            agg_fwd_time += forward_time.count();
            agg_bwd_time += backward_time.count();
            agg_batch_time += batch_time.count();
            ++agg_count;

            // Periodic evaluation on small validation batch
            if ((num_batches % eval_interval) == 0) {
                torch::Tensor v_images, v_labels;
                if (!val_set.next_batch(batch_size, device_, v_images, v_labels)) {
                    val_set.reset(/*shuffle=*/false);
                    val_set.next_batch(batch_size, device_, v_images, v_labels);
                }

                // Compute evaluation metrics on this validation batch (no grad)
                {
                    torch::NoGradGuard no_grad;

                    auto v_recon = autoencoder_.forward(v_images);

                    // MSE (mean over all elements)
                    auto mse_t = mse_loss_cpu(v_recon, v_images);
                    double mse = mse_t.item<double>();

                    // MAE
                    auto mae_t = torch::l1_loss(v_recon, v_images, torch::Reduction::Mean);
                    double mae = mae_t.item<double>();

                    // Per-channel MSE: mean over N,H,W -> shape [C]
                    auto per_ch_mse = (v_recon - v_images).pow(2).mean({0, 2, 3});

                    // PSNR (assuming pixel range [0,1])
                    double psnr = (mse <= 0.0) ? std::numeric_limits<double>::infinity()
                                                : 10.0 * std::log10(1.0 / mse);

                    // Absolute error percentiles (median, 90th)
                    auto absdiff = (v_recon - v_images).abs().view(-1);
                    auto sorted = std::get<0>(torch::sort(absdiff));
                    int64_t nvals = sorted.size(0);
                    double median = 0.0, p90 = 0.0;
                    if (nvals > 0) {
                        int64_t idx50 = nvals / 2;
                        int64_t idx90 = std::min<int64_t>(nvals - 1, static_cast<int64_t>(std::round(0.9 * (nvals - 1))));
                        median = sorted[idx50].item<double>();
                        p90 = sorted[idx90].item<double>();
                    }

                    // Print concise metric summary
                    std::cout << "  [Epoch " << (epoch + 1) << ", Batch " << num_batches
                              << "] Eval (val batch) MSE: " << std::fixed << std::setprecision(6) << mse
                              << " | MAE: " << std::setprecision(6) << mae
                              << " | PSNR: " << std::setprecision(3) << psnr << " dB"
                              << " | MedianErr: " << std::setprecision(6) << median
                              << " | P90Err: " << std::setprecision(6) << p90
                              << std::endl;

                    // Print per-channel MSE
                    auto pcm_cpu = per_ch_mse.to(torch::kCPU);
                    std::cout << "    Per-channel MSE: [";
                    for (int c = 0; c < pcm_cpu.size(0); ++c) {
                        std::cout << std::fixed << std::setprecision(6) << pcm_cpu[c].item<double>();
                        if (c + 1 < pcm_cpu.size(0)) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            }

            // Periodic aggregated logging every `log_interval` batches
            if ((num_batches % log_interval) == 0) {
                double avg_loss = agg_loss / std::max(1, agg_count);
                double avg_fwd = agg_fwd_time / std::max(1, agg_count);
                double avg_bwd = agg_bwd_time / std::max(1, agg_count);
                double avg_batch = agg_batch_time / std::max(1, agg_count);
                double progress = (100.0 * num_batches) / batches_per_epoch;

                std::cout << "  [Epoch " << (epoch + 1) << "] "
                          << "Batches " << (num_batches - agg_count + 1) << "-" << num_batches
                          << " | Avg Loss: " << std::fixed << std::setprecision(6) << avg_loss
                          << " | Avg Fwd: " << std::setprecision(3) << avg_fwd << "s"
                          << " | Avg Bwd: " << std::setprecision(3) << avg_bwd << "s"
                          << " | Avg Total: " << std::setprecision(3) << avg_batch << "s"
                          << " | Progress: " << std::setprecision(1) << progress << "%"
                          << std::endl;

                // reset aggregation
                agg_loss = agg_fwd_time = agg_bwd_time = agg_batch_time = 0.0;
                agg_count = 0;
            }
        }

        auto epoch_end   = Clock::now();
        std::chrono::duration<double> elapsed = epoch_end - epoch_start;
        double avg_loss = epoch_loss_sum / std::max(1, num_batches);

        std::cout << "===== Epoch [" << (epoch + 1) << "/" << num_epochs << "] COMPLETE "
                  << "===== Avg Loss: " << std::fixed << std::setprecision(6) << avg_loss
                  << " - Total Time: " << std::setprecision(2) << elapsed.count() << "s "
                  << "(" << std::setprecision(2) << (elapsed.count() / 60.0) << " min)"
                  << std::endl << std::endl;
    }

    // Save autoencoder sau khi train
    autoencoder_.save(ae_save_path);
    std::cout << "[INFO] Autoencoder saved to: " << ae_save_path << std::endl;
}

// ===== Load AE đã train =====
void TrainingPipeline::load_autoencoder(const std::string& ae_path) {
    autoencoder_.load(ae_path);
    std::cout << "[INFO] Loaded autoencoder from: " << ae_path << std::endl;
}

// ===== Phase 2: Extract features (train/test) =====
void TrainingPipeline::extract_features(bool                              train,
                                        int                               batch_size,
                                        std::vector<std::vector<double>>& features,
                                        std::vector<int>&                 labels)
{
    std::cout << "===== EXTRACT FEATURES ("
              << (train ? "TRAIN" : "TEST") << " SET) =====" << std::endl;

    CIFAR10Dataset dataset(cifar_root_, /*train=*/train);
    dataset.reset(/*shuffle=*/false);

    torch::Tensor images, tlabels;

    torch::NoGradGuard no_grad;

    while (dataset.next_batch(batch_size, device_, images, tlabels)) {
        // encoder output: [B, 128, 8, 8]
        auto latent = autoencoder_.extract_features(images);
        latent = latent.view({latent.size(0), -1});  // [B, F]

        const auto B = latent.size(0);
        const auto F = latent.size(1);

        auto latent_cpu = latent.to(torch::kCPU);
        auto labels_cpu = tlabels.to(torch::kCPU);

        for (int i = 0; i < B; ++i) {
            std::vector<double> feat(F);
            for (int j = 0; j < F; ++j) {
                feat[j] = latent_cpu[i][j].item<double>();
            }
            features.push_back(std::move(feat));

            int y = static_cast<int>(labels_cpu[i].item<int64_t>());
            labels.push_back(y);
        }
    }

    std::cout << "[INFO] Extracted " << features.size()
              << " feature vectors, dim = " 
              << (features.empty() ? 0 : features[0].size())
              << std::endl;
}

} // namespace dl
