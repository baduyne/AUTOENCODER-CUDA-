#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include <memory>

#include "dl/autoencoder_cpu.h"
#include "dl/cifar10_dataset.h"

namespace dl {

class TrainingPipelineCPU {
public:
    explicit TrainingPipelineCPU(const std::string& cifar_root);

    void train_autoencoder(int num_epochs,
                          int batch_size,
                          float learning_rate,
                          const std::string& ae_save_path);

    // Extract latent features for SVM training. Fills `out_features` with
    // one vector<double> per sample (flattened latent of size 8192), and
    // `out_labels` with corresponding labels.
    void extract_features(bool train,
                          int batch_size,
                          std::vector<std::vector<double>>& out_features,
                          std::vector<int>& out_labels);

    // Extract and save features to binary files (for SVM phase)
    // Saves: features_path (float32, row-major) and labels_path (int32)
    void extract_and_save_features(bool train,
                                    int batch_size,
                                    int max_samples,
                                    const std::string& features_path,
                                    const std::string& labels_path);

    void load_autoencoder(const std::string& path) { autoencoder_->load(path); }

    // TODO: extract_features, evaluate... nếu cần về sau

    AutoencoderCPU&       autoencoder()       { return *autoencoder_; }
    const AutoencoderCPU& autoencoder() const { return *autoencoder_; }

private:
    std::string           cifar_root_;
    std::unique_ptr<AutoencoderCPU> autoencoder_; // phase autoencoder sẽ refactor tiếp
};

} // namespace dl

