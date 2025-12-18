#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>

#include "dl/autoencoder.h"
#include "dl/cifar10_dataset.h"

namespace dl {

class TrainingPipeline {
public:
    TrainingPipeline(const std::string& cifar_root, torch::Device device);

    // Phase 1: train autoencoder
    void train_autoencoder(int   num_epochs,
                           int   batch_size,
                           float learning_rate,
                           const std::string& ae_save_path);

    // Load autoencoder đã train (để extract feature ở phase sau)
    void load_autoencoder(const std::string& ae_path);

    // Phase 2: extract features từ encoder cho train/test
    // train = true  -> CIFAR-10 train set
    // train = false -> CIFAR-10 test set
    void extract_features(bool                                train,
                          int                                 batch_size,
                          std::vector<std::vector<double>>&   features,
                          std::vector<int>&                   labels);

    Autoencoder&       autoencoder()       { return autoencoder_; }
    const Autoencoder& autoencoder() const { return autoencoder_; }

private:
    std::string   cifar_root_;
    torch::Device device_;
    Autoencoder   autoencoder_;
};

} // namespace dl
