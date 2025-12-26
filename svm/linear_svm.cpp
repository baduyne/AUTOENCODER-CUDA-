#include "linear_svm.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <cmath>

namespace svm {

// =============================================================================
// LinearSVMTrainer Implementation
// =============================================================================

struct model* LinearSVMTrainer::train(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    const Config& config)
{
    if (features.empty() || labels.empty()) {
        throw std::runtime_error("[LinearSVM] Empty features or labels");
    }
    if (features.size() != labels.size()) {
        throw std::runtime_error("[LinearSVM] Feature/label size mismatch");
    }

    const int num_samples = static_cast<int>(features.size());
    const int num_features = static_cast<int>(features[0].size());

    std::cout << "[LinearSVM] Training with " << num_samples << " samples, "
              << num_features << " features" << std::endl;
    std::cout << "[LinearSVM] Config: C=" << config.C
              << ", eps=" << config.eps
              << ", solver=" << config.solver_type
              << ", bias=" << config.bias << std::endl;

    // Build problem structure
    struct problem prob;
    prob.l = num_samples;
    prob.n = num_features;
    prob.bias = config.bias;

    // Allocate labels
    prob.y = new double[num_samples];
    for (int i = 0; i < num_samples; ++i) {
        prob.y[i] = static_cast<double>(labels[i]);
    }

    // Allocate feature nodes
    // Each sample needs (num_features + 1) nodes for bias, plus sentinel
    int nodes_per_sample = num_features + (config.bias >= 0 ? 1 : 0) + 1;
    struct feature_node* x_space = new struct feature_node[num_samples * nodes_per_sample];
    prob.x = new struct feature_node*[num_samples];

    for (int i = 0; i < num_samples; ++i) {
        prob.x[i] = &x_space[i * nodes_per_sample];
        int idx = 0;

        // Copy features (1-indexed for liblinear)
        for (int j = 0; j < num_features; ++j) {
            prob.x[i][idx].index = j + 1;
            prob.x[i][idx].value = features[i][j];
            ++idx;
        }

        // Add bias term if enabled
        if (config.bias >= 0) {
            prob.x[i][idx].index = num_features + 1;
            prob.x[i][idx].value = config.bias;
            ++idx;
        }

        // Sentinel
        prob.x[i][idx].index = -1;
        prob.x[i][idx].value = 0;
    }

    // Set parameters
    struct parameter param;
    param.solver_type = config.solver_type;
    param.C = config.C;
    param.eps = config.eps;
    param.p = 0.1;
    param.nu = 0.5;
    param.nr_weight = 0;
    param.weight_label = nullptr;
    param.weight = nullptr;
    param.init_sol = nullptr;
    param.regularize_bias = 1;

    // Check parameters
    const char* error_msg = check_parameter(&prob, &param);
    if (error_msg) {
        delete[] prob.y;
        delete[] prob.x;
        delete[] x_space;
        throw std::runtime_error(std::string("[LinearSVM] Parameter error: ") + error_msg);
    }

    // Train
    std::cout << "[LinearSVM] Training started..." << std::endl;
    struct model* m = ::train(&prob, &param);

    if (!m) {
        delete[] prob.y;
        delete[] prob.x;
        delete[] x_space;
        throw std::runtime_error("[LinearSVM] Training failed");
    }

    std::cout << "[LinearSVM] Training completed. Classes: " << m->nr_class
              << ", Features: " << m->nr_feature << std::endl;

    // Cleanup problem structure (model has its own copy)
    delete[] prob.y;
    delete[] prob.x;
    delete[] x_space;

    return m;
}

struct model* LinearSVMTrainer::train(
    const std::vector<std::vector<float>>& features,
    const std::vector<int>& labels,
    const Config& config)
{
    // Convert float to double
    std::vector<std::vector<double>> double_features(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
        double_features[i].resize(features[i].size());
        for (size_t j = 0; j < features[i].size(); ++j) {
            double_features[i][j] = static_cast<double>(features[i][j]);
        }
    }
    return train(double_features, labels, config);
}

bool LinearSVMTrainer::save_model(const struct model* m, const std::string& path) {
    if (!m) {
        std::cerr << "[LinearSVM] Cannot save null model" << std::endl;
        return false;
    }
    int result = ::save_model(path.c_str(), m);
    if (result == 0) {
        std::cout << "[LinearSVM] Model saved to: " << path << std::endl;
        return true;
    }
    std::cerr << "[LinearSVM] Failed to save model to: " << path << std::endl;
    return false;
}

struct model* LinearSVMTrainer::load_model(const std::string& path) {
    struct model* m = ::load_model(path.c_str());
    if (m) {
        std::cout << "[LinearSVM] Model loaded from: " << path << std::endl;
        std::cout << "[LinearSVM] Classes: " << m->nr_class
                  << ", Features: " << m->nr_feature << std::endl;
    } else {
        std::cerr << "[LinearSVM] Failed to load model from: " << path << std::endl;
    }
    return m;
}

void LinearSVMTrainer::free_model(struct model*& m) {
    if (m) {
        free_and_destroy_model(&m);
        m = nullptr;
    }
}

// =============================================================================
// LinearSVMEvaluator Implementation
// =============================================================================

EvalResult LinearSVMEvaluator::evaluate(
    const struct model* m,
    const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    int num_classes)
{
    EvalResult result;
    result.total_samples = static_cast<int>(features.size());
    result.correct_predictions = 0;
    result.confusion_matrix.resize(num_classes, std::vector<int>(num_classes, 0));

    if (!m || features.empty()) {
        return result;
    }

    const int num_features = static_cast<int>(features[0].size());

    // Count class distribution for baseline
    std::vector<int> class_counts(num_classes, 0);
    for (int label : labels) {
        if (label >= 0 && label < num_classes) {
            class_counts[label]++;
        }
    }
    int max_count = *std::max_element(class_counts.begin(), class_counts.end());
    result.baseline_accuracy = static_cast<double>(max_count) / result.total_samples;

    // Allocate feature nodes for prediction
    std::vector<struct feature_node> x(num_features + 2);

    std::cout << "[LinearSVM] Evaluating " << result.total_samples << " samples..." << std::endl;

    for (size_t i = 0; i < features.size(); ++i) {
        // Build feature vector
        for (int j = 0; j < num_features; ++j) {
            x[j].index = j + 1;
            x[j].value = features[i][j];
        }
        x[num_features].index = -1;
        x[num_features].value = 0;

        // Predict
        int pred = static_cast<int>(::predict(m, x.data()));
        int true_label = labels[i];

        // Update confusion matrix
        if (true_label >= 0 && true_label < num_classes &&
            pred >= 0 && pred < num_classes) {
            result.confusion_matrix[true_label][pred]++;
        }

        if (pred == true_label) {
            result.correct_predictions++;
        }

        // Progress indicator
        if ((i + 1) % 1000 == 0) {
            std::cout << "\r[LinearSVM] Evaluated " << (i + 1) << "/" << result.total_samples << std::flush;
        }
    }
    std::cout << "\r[LinearSVM] Evaluated " << result.total_samples << "/" << result.total_samples << std::endl;

    result.accuracy = static_cast<double>(result.correct_predictions) / result.total_samples;
    return result;
}

EvalResult LinearSVMEvaluator::evaluate(
    const struct model* m,
    const std::vector<std::vector<float>>& features,
    const std::vector<int>& labels,
    int num_classes)
{
    // Convert float to double
    std::vector<std::vector<double>> double_features(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
        double_features[i].resize(features[i].size());
        for (size_t j = 0; j < features[i].size(); ++j) {
            double_features[i][j] = static_cast<double>(features[i][j]);
        }
    }
    return evaluate(m, double_features, labels, num_classes);
}

int LinearSVMEvaluator::predict(const struct model* m, const std::vector<double>& features) {
    if (!m || features.empty()) return -1;

    std::vector<struct feature_node> x(features.size() + 1);
    for (size_t j = 0; j < features.size(); ++j) {
        x[j].index = static_cast<int>(j + 1);
        x[j].value = features[j];
    }
    x[features.size()].index = -1;
    x[features.size()].value = 0;

    return static_cast<int>(::predict(m, x.data()));
}

int LinearSVMEvaluator::predict(const struct model* m, const std::vector<float>& features) {
    std::vector<double> double_features(features.begin(), features.end());
    return predict(m, double_features);
}

void LinearSVMEvaluator::print_results(const EvalResult& result) {
    std::cout << "\n========================================\n"
              << "         EVALUATION RESULTS\n"
              << "========================================\n"
              << "Total samples:      " << result.total_samples << "\n"
              << "Correct predictions: " << result.correct_predictions << "\n"
              << "Accuracy:           " << std::fixed << std::setprecision(4)
              << (result.accuracy * 100.0) << "%\n"
              << "Baseline (majority): " << std::fixed << std::setprecision(4)
              << (result.baseline_accuracy * 100.0) << "%\n"
              << "Improvement:        " << std::fixed << std::setprecision(4)
              << ((result.accuracy - result.baseline_accuracy) * 100.0) << "%\n"
              << "========================================\n" << std::endl;
}

void LinearSVMEvaluator::print_confusion_matrix(const EvalResult& result) {
    int num_classes = static_cast<int>(result.confusion_matrix.size());
    if (num_classes == 0) return;

    std::cout << "\nConfusion Matrix:\n";
    std::cout << "      ";
    for (int j = 0; j < num_classes; ++j) {
        std::cout << std::setw(6) << j;
    }
    std::cout << "\n";

    for (int i = 0; i < num_classes; ++i) {
        std::cout << std::setw(4) << i << ": ";
        for (int j = 0; j < num_classes; ++j) {
            std::cout << std::setw(6) << result.confusion_matrix[i][j];
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// =============================================================================
// Utility Functions
// =============================================================================

namespace utils {

NormalizationStats compute_normalization_stats(
    const std::vector<std::vector<double>>& features)
{
    if (features.empty()) {
        throw std::runtime_error("[LinearSVM] Cannot compute stats from empty features");
    }

    const int num_samples = static_cast<int>(features.size());
    const int feature_dim = static_cast<int>(features[0].size());

    NormalizationStats stats;
    stats.feature_dim = feature_dim;
    stats.mean.resize(feature_dim, 0.0);
    stats.std_dev.resize(feature_dim, 0.0);

    // Compute mean
    for (const auto& sample : features) {
        for (int j = 0; j < feature_dim; ++j) {
            stats.mean[j] += sample[j];
        }
    }
    for (int j = 0; j < feature_dim; ++j) {
        stats.mean[j] /= num_samples;
    }

    // Compute standard deviation
    for (const auto& sample : features) {
        for (int j = 0; j < feature_dim; ++j) {
            double diff = sample[j] - stats.mean[j];
            stats.std_dev[j] += diff * diff;
        }
    }
    for (int j = 0; j < feature_dim; ++j) {
        stats.std_dev[j] = std::sqrt(stats.std_dev[j] / num_samples);
        // Prevent division by zero for constant features
        if (stats.std_dev[j] < 1e-10) {
            stats.std_dev[j] = 1.0;
        }
    }

    std::cout << "[LinearSVM] Normalization stats computed: feature_dim=" << feature_dim << std::endl;
    return stats;
}

void normalize_features(
    std::vector<std::vector<double>>& features,
    const NormalizationStats& stats)
{
    if (features.empty()) return;

    const int feature_dim = static_cast<int>(features[0].size());
    if (feature_dim != stats.feature_dim) {
        throw std::runtime_error("[LinearSVM] Feature dimension mismatch in normalization");
    }

    for (auto& sample : features) {
        for (int j = 0; j < feature_dim; ++j) {
            sample[j] = (sample[j] - stats.mean[j]) / stats.std_dev[j];
        }
    }

    std::cout << "[LinearSVM] Normalized " << features.size() << " samples" << std::endl;
}

NormalizationStats normalize_features_inplace(
    std::vector<std::vector<double>>& features)
{
    NormalizationStats stats = compute_normalization_stats(features);
    normalize_features(features, stats);
    return stats;
}

bool load_features_bin(
    const std::string& path,
    std::vector<std::vector<double>>& features,
    int feature_dim)
{
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[LinearSVM] Cannot open feature file: " << path << std::endl;
        return false;
    }

    fin.seekg(0, std::ios::end);
    std::streamsize file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    size_t num_samples = file_size / (feature_dim * sizeof(float));
    if (file_size % (feature_dim * sizeof(float)) != 0) {
        std::cerr << "[LinearSVM] Warning: file size not aligned with feature_dim" << std::endl;
    }

    std::cout << "[LinearSVM] Loading " << num_samples << " samples from " << path << std::endl;

    features.clear();
    features.reserve(num_samples);

    std::vector<float> buffer(feature_dim);
    for (size_t i = 0; i < num_samples; ++i) {
        fin.read(reinterpret_cast<char*>(buffer.data()), feature_dim * sizeof(float));
        if (!fin) {
            std::cerr << "[LinearSVM] Failed to read sample " << i << std::endl;
            return false;
        }

        std::vector<double> sample(feature_dim);
        for (int j = 0; j < feature_dim; ++j) {
            sample[j] = static_cast<double>(buffer[j]);
        }
        features.push_back(std::move(sample));
    }

    fin.close();
    return true;
}

bool load_labels_bin(const std::string& path, std::vector<int>& labels) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[LinearSVM] Cannot open label file: " << path << std::endl;
        return false;
    }

    fin.seekg(0, std::ios::end);
    std::streamsize file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    size_t num_samples = file_size / sizeof(int32_t);

    std::cout << "[LinearSVM] Loading " << num_samples << " labels from " << path << std::endl;

    labels.clear();
    labels.reserve(num_samples);

    int32_t label;
    for (size_t i = 0; i < num_samples; ++i) {
        fin.read(reinterpret_cast<char*>(&label), sizeof(int32_t));
        if (!fin) {
            std::cerr << "[LinearSVM] Failed to read label " << i << std::endl;
            return false;
        }
        labels.push_back(static_cast<int>(label));
    }

    fin.close();
    return true;
}

} // namespace utils

} // namespace svm
