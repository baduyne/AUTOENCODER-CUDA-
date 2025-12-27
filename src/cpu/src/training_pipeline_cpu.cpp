#include "dl/training_pipeline_cpu.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <sys/resource.h>
#include <fstream>

namespace dl {

using Clock = std::chrono::high_resolution_clock;

TrainingPipelineCPU::TrainingPipelineCPU(const std::string& cifar_root)
    : cifar_root_(cifar_root) {
    autoencoder_.reset(new AutoencoderCPU());
}

void TrainingPipelineCPU::train_autoencoder(int num_epochs, int batch_size, float learning_rate, const std::string& ae_save_path) {
    std::cout << "===== TRAINING AUTOENCODER (CPU, float*) =====" << std::endl;
    try {
        const size_t max_samples = 500; // limit to 500 images as requested
        CIFAR10Dataset train_set(cifar_root_, true, max_samples);
        std::cout << "Train size: " << train_set.size() << " samples" << std::endl;
        
        if (train_set.size() == 0) {
            std::cerr << "[ERROR] Dataset is empty! Please check if CIFAR-10 files exist at: " << cifar_root_ << std::endl;
            return;
        }
        
        const int total_samples = train_set.size();
        const int batches_per_epoch = (total_samples + batch_size - 1) / batch_size;
        
        std::cout << "Total samples: " << total_samples 
                  << ", Batches per epoch: " << batches_per_epoch 
                  << ", Batch size: " << batch_size << std::endl;
        
        // Prepare memory log CSV
        const std::string mem_csv = "memory_log.csv";
        const std::string epoch_csv = "epoch_log.csv";
        // if epoch log doesn't exist, write header
        {
            std::ifstream ifs(epoch_csv);
            if (!ifs.good()) {
                std::ofstream ofs(epoch_csv);
                ofs << "epoch,train_loss,eval_loss,avg_time_s,peak_kb\n";
            }
        }
        // if file doesn't exist, write header
        {
            std::ifstream ifs(mem_csv);
            if (!ifs.good()) {
                std::ofstream ofs(mem_csv);
                ofs << "phase,stage,epoch,peak_kb,avg_loss,elapsed_s\n";
            }
        }

        long prev_maxrss_kb = 0; // baseline ru_maxrss (KB)

        // enforce requested settings: 500 images, batch size 32, 20 epochs
        num_epochs = 20;
        batch_size = 32;
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // sample baseline maxrss before epoch
            struct rusage ru_before; getrusage(RUSAGE_SELF, &ru_before);
            prev_maxrss_kb = ru_before.ru_maxrss; // kilobytes

            auto epoch_start = Clock::now();
            train_set.reset(true);
            double epoch_loss_sum = 0.0;
            int    num_batches    = 0;

            while (true) {
                CIFAR10Batch batch;
                if (!train_set.next_batch(batch_size, batch)) break;

                auto batch_start = Clock::now();
                // Forward (float*)
                float* recon = new float[batch.batch_size * 3 * 32 * 32]; // output buffer, sẽ cấp phát/fill thực tế trong autoencoder sau
                autoencoder_->forward(batch.images, recon, batch.batch_size); // interface sẽ update khi phase autoencoder refactor
                // Loss
                float batch_loss = autoencoder_->reconstruction_loss(batch.images, recon, batch.batch_size);
                epoch_loss_sum += batch_loss;
                ++num_batches;
                // Backward + update
                autoencoder_->backward(batch.images, recon, learning_rate, batch.batch_size);
                auto batch_end = Clock::now();
                std::chrono::duration<double> batch_time = batch_end - batch_start;

                // no per-20-batch logging; we will log per-epoch instead
                delete[] recon;
                // Batch pointers will be automatically cleaned up by destructor
            }
                    // Example: extraction API usage (not executed here)
                    // std::vector<std::vector<double>> feats; std::vector<int> labels;
                    // pipeline.extract_features(true, batch_size, feats, labels);
            auto epoch_end   = Clock::now();
            std::chrono::duration<double> elapsed = epoch_end - epoch_start;
            double avg_loss = epoch_loss_sum / std::max(1, num_batches);
            // sample ru after epoch to compute peak increase
            struct rusage ru_after; getrusage(RUSAGE_SELF, &ru_after);
            long after_maxrss_kb = ru_after.ru_maxrss;
            long peak_kb = 0;
            if (after_maxrss_kb > prev_maxrss_kb) peak_kb = after_maxrss_kb - prev_maxrss_kb;

            std::cout << "===== Epoch [" << (epoch+1) << "/" << num_epochs << "] COMPLETE "
                      << "===== Avg Loss: " << std::fixed << std::setprecision(6) << avg_loss
                      << " - Total Time: " << std::setprecision(2) << elapsed.count() << "s "
                      << "(" << std::setprecision(2) << (elapsed.count() / 60.0) << " min)"
                      << " | Mem peak (KB): " << peak_kb
                      << std::endl << std::endl;

            // compute eval loss on limited validation (test) set
            CIFAR10Dataset val_set(cifar_root_, false, max_samples);
            val_set.reset(false);
            double eval_loss_sum = 0.0;
            int eval_batches = 0;
            while (true) {
                CIFAR10Batch batch;
                if (!val_set.next_batch(batch_size, batch)) break;
                float* recon_v = new float[batch.batch_size * 3 * 32 * 32];
                autoencoder_->forward(batch.images, recon_v, batch.batch_size);
                float b_loss = autoencoder_->reconstruction_loss(batch.images, recon_v, batch.batch_size);
                eval_loss_sum += b_loss;
                ++eval_batches;
                delete[] recon_v;
            }
            double avg_eval_loss = eval_batches > 0 ? (eval_loss_sum / eval_batches) : 0.0;

            // append memory.csv (existing) and new epoch-level csv
            std::ofstream ofs_mem(mem_csv, std::ios::app);
            if (ofs_mem) {
                ofs_mem << "autoencoder,train," << (epoch+1) << "," << peak_kb << "," << avg_loss << "," << elapsed.count() << "\n";
            }
            std::ofstream ofs(epoch_csv, std::ios::app);
            if (ofs) {
                ofs << (epoch+1) << "," << avg_loss << "," << avg_eval_loss << "," << elapsed.count() << "," << peak_kb << "\n";
            }
        }
        // Save weight (chờ phase autoencoder refactor tiếp)
        autoencoder_->save(ae_save_path);
        std::cout << "[INFO] Autoencoder saved to: " << ae_save_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during training: " << e.what() << std::endl;
        std::cerr << "[ERROR] Please check if CIFAR-10 dataset files exist at: " << cifar_root_ << std::endl;
        throw;
    }
}

void TrainingPipelineCPU::extract_features(bool train, int batch_size,
                                          std::vector<std::vector<double>>& out_features,
                                          std::vector<int>& out_labels) {
    CIFAR10Dataset dataset(cifar_root_, train, 500);
    out_features.clear(); out_labels.clear();
    dataset.reset(false); // deterministic order by default

    size_t latent_per = AutoencoderCPU::latent_size();
    std::vector<float> latent_buf;

    while (true) {
        CIFAR10Batch batch;
        if (!dataset.next_batch(batch_size, batch)) break;
        // prepare buffer
        latent_buf.resize(batch.batch_size * latent_per);
        autoencoder_->extract_latent(batch.images, batch.batch_size, latent_buf.data());

        for (size_t i = 0; i < batch.batch_size; ++i) {
            const float* src = latent_buf.data() + i * latent_per;
            std::vector<double> feat(latent_per);
            for (size_t k = 0; k < latent_per; ++k) feat[k] = static_cast<double>(src[k]);
            out_features.emplace_back(std::move(feat));
            out_labels.emplace_back(static_cast<int>(batch.labels[i]));
        }
        // batch destructor frees memory
    }
}

void TrainingPipelineCPU::extract_and_save_features(bool train,
                                                     int batch_size,
                                                     int max_samples,
                                                     const std::string& features_path,
                                                     const std::string& labels_path) {
    std::cout << "\n========== EXTRACT " << (train ? "TRAIN" : "TEST") << " FEATURES (CPU) ==========" << std::endl;

    CIFAR10Dataset dataset(cifar_root_, train, max_samples);
    dataset.reset(false); // deterministic order

    const size_t LATENT_DIM = AutoencoderCPU::latent_size(); // 8192
    std::vector<float> all_features;
    std::vector<int32_t> all_labels;

    std::vector<float> latent_buf;
    size_t total_extracted = 0;

    while (true) {
        CIFAR10Batch batch;
        if (!dataset.next_batch(batch_size, batch)) break;

        // Extract latent features
        latent_buf.resize(batch.batch_size * LATENT_DIM);
        autoencoder_->extract_latent(batch.images, batch.batch_size, latent_buf.data());

        // Append to vectors
        all_features.insert(all_features.end(), latent_buf.begin(), latent_buf.end());
        for (size_t i = 0; i < batch.batch_size; ++i) {
            all_labels.push_back(static_cast<int32_t>(batch.labels[i]));
        }

        total_extracted += batch.batch_size;
        std::cout << "\rExtracted " << total_extracted << " / " << max_samples << " images" << std::flush;
    }
    std::cout << std::endl;

    // Save features (float32, row-major)
    std::ofstream feat_file(features_path, std::ios::binary);
    if (!feat_file) {
        throw std::runtime_error("Cannot open features file: " + features_path);
    }
    feat_file.write(reinterpret_cast<const char*>(all_features.data()),
                    all_features.size() * sizeof(float));
    feat_file.close();
    std::cout << "[CPU] Saved features to: " << features_path << std::endl;

    // Save labels (int32)
    std::ofstream label_file(labels_path, std::ios::binary);
    if (!label_file) {
        throw std::runtime_error("Cannot open labels file: " + labels_path);
    }
    label_file.write(reinterpret_cast<const char*>(all_labels.data()),
                     all_labels.size() * sizeof(int32_t));
    label_file.close();
    std::cout << "[CPU] Saved labels to: " << labels_path << std::endl;

    std::cout << "[CPU] Total samples: " << total_extracted << std::endl;
}

} // namespace dl

