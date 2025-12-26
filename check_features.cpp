/**
 * Diagnostic tool to check feature files for issues
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdint>
#include <limits>

const int FEATURE_DIM = 8192;

struct FeatureStats {
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    double sum = 0;
    double sum_sq = 0;
    int num_zeros = 0;
    int num_nan = 0;
    int num_inf = 0;
    int total_values = 0;
};

void analyze_features(const std::string& path, int max_samples = 1000) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Cannot open: " << path << std::endl;
        return;
    }

    fin.seekg(0, std::ios::end);
    size_t file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    size_t num_samples = file_size / (FEATURE_DIM * sizeof(float));
    std::cout << "\n=== Analyzing: " << path << " ===\n";
    std::cout << "File size: " << file_size << " bytes\n";
    std::cout << "Total samples: " << num_samples << "\n";

    FeatureStats stats;
    std::vector<float> buffer(FEATURE_DIM);

    // Per-feature stats
    std::vector<double> feat_min(FEATURE_DIM, std::numeric_limits<double>::max());
    std::vector<double> feat_max(FEATURE_DIM, std::numeric_limits<double>::lowest());
    std::vector<double> feat_sum(FEATURE_DIM, 0);

    int samples_to_check = std::min((int)num_samples, max_samples);

    for (int i = 0; i < samples_to_check; ++i) {
        fin.read(reinterpret_cast<char*>(buffer.data()), FEATURE_DIM * sizeof(float));

        for (int j = 0; j < FEATURE_DIM; ++j) {
            float val = buffer[j];
            stats.total_values++;

            if (std::isnan(val)) {
                stats.num_nan++;
                continue;
            }
            if (std::isinf(val)) {
                stats.num_inf++;
                continue;
            }
            if (val == 0.0f) {
                stats.num_zeros++;
            }

            stats.min_val = std::min(stats.min_val, (double)val);
            stats.max_val = std::max(stats.max_val, (double)val);
            stats.sum += val;
            stats.sum_sq += val * val;

            feat_min[j] = std::min(feat_min[j], (double)val);
            feat_max[j] = std::max(feat_max[j], (double)val);
            feat_sum[j] += val;
        }
    }

    double mean = stats.sum / stats.total_values;
    double variance = (stats.sum_sq / stats.total_values) - (mean * mean);
    double std_dev = std::sqrt(std::max(0.0, variance));

    std::cout << "\nGlobal Statistics (first " << samples_to_check << " samples):\n";
    std::cout << "  Min value:    " << std::fixed << std::setprecision(6) << stats.min_val << "\n";
    std::cout << "  Max value:    " << stats.max_val << "\n";
    std::cout << "  Mean:         " << mean << "\n";
    std::cout << "  Std dev:      " << std_dev << "\n";
    std::cout << "  Zero values:  " << stats.num_zeros << " ("
              << (100.0 * stats.num_zeros / stats.total_values) << "%)\n";
    std::cout << "  NaN values:   " << stats.num_nan << "\n";
    std::cout << "  Inf values:   " << stats.num_inf << "\n";

    // Show first few samples
    fin.seekg(0, std::ios::beg);
    std::cout << "\nFirst 3 samples (first 10 features each):\n";
    for (int i = 0; i < 3 && i < (int)num_samples; ++i) {
        fin.read(reinterpret_cast<char*>(buffer.data()), FEATURE_DIM * sizeof(float));
        std::cout << "  Sample " << i << ": [";
        for (int j = 0; j < 10; ++j) {
            std::cout << std::fixed << std::setprecision(4) << buffer[j];
            if (j < 9) std::cout << ", ";
        }
        std::cout << ", ...]\n";
    }

    // Check feature variance
    int zero_variance_features = 0;
    for (int j = 0; j < FEATURE_DIM; ++j) {
        if (feat_min[j] == feat_max[j]) {
            zero_variance_features++;
        }
    }
    std::cout << "\nFeatures with zero variance: " << zero_variance_features
              << " / " << FEATURE_DIM << "\n";

    fin.close();
}

void analyze_labels(const std::string& path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Cannot open: " << path << std::endl;
        return;
    }

    fin.seekg(0, std::ios::end);
    size_t file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    size_t num_samples = file_size / sizeof(int32_t);
    std::cout << "\n=== Analyzing: " << path << " ===\n";
    std::cout << "Total labels: " << num_samples << "\n";

    std::vector<int> class_counts(10, 0);
    int32_t label;
    int min_label = INT32_MAX, max_label = INT32_MIN;

    for (size_t i = 0; i < num_samples; ++i) {
        fin.read(reinterpret_cast<char*>(&label), sizeof(int32_t));
        min_label = std::min(min_label, label);
        max_label = std::max(max_label, label);
        if (label >= 0 && label < 10) {
            class_counts[label]++;
        }
    }

    std::cout << "Label range: [" << min_label << ", " << max_label << "]\n";
    std::cout << "\nClass distribution:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "  Class " << i << ": " << class_counts[i]
                  << " (" << std::fixed << std::setprecision(2)
                  << (100.0 * class_counts[i] / num_samples) << "%)\n";
    }

    fin.close();
}

int main() {
    std::cout << "========================================\n"
              << "   Feature File Diagnostic Tool\n"
              << "========================================\n";

    analyze_features("train_features.bin", 1000);
    analyze_labels("train_labels.bin");

    analyze_features("test_features.bin", 1000);
    analyze_labels("test_labels.bin");

    return 0;
}
