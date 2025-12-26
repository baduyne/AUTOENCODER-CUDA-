#ifndef LINEAR_SVM_H
#define LINEAR_SVM_H

#include <vector>
#include <string>
#include "linear.h"

namespace svm {

// =============================================================================
// Configuration for Linear SVM
// =============================================================================
struct Config {
    double C = 1.0;                          // Regularization parameter
    double eps = 0.1;                        // Stopping tolerance
    int solver_type = L2R_L2LOSS_SVC_DUAL;   // Default: L2-regularized L2-loss SVC (dual)
    double bias = -1.0;                      // Bias term (-1 = no bias)

    // Solver types available:
    // L2R_LR               = 0  (L2-regularized logistic regression)
    // L2R_L2LOSS_SVC_DUAL  = 1  (L2-regularized L2-loss SVC dual) - DEFAULT
    // L2R_L2LOSS_SVC       = 2  (L2-regularized L2-loss SVC primal)
    // L2R_L1LOSS_SVC_DUAL  = 3  (L2-regularized L1-loss SVC dual)
    // MCSVM_CS             = 4  (Crammer-Singer multi-class SVM)
    // L1R_L2LOSS_SVC       = 5  (L1-regularized L2-loss SVC)
    // L1R_LR               = 6  (L1-regularized logistic regression)
    // L2R_LR_DUAL          = 7  (L2-regularized logistic regression dual)
};

// =============================================================================
// Evaluation result structure
// =============================================================================
struct EvalResult {
    double accuracy = 0.0;
    std::vector<std::vector<int>> confusion_matrix;
    double baseline_accuracy = 0.0;  // Majority class accuracy
    int total_samples = 0;
    int correct_predictions = 0;
};

// =============================================================================
// Linear SVM Trainer - wraps liblinear for training
// =============================================================================
class LinearSVMTrainer {
public:
    // Train a linear SVM model
    // features: [num_samples][num_features] - feature vectors (double)
    // labels: [num_samples] - class labels
    // config: SVM configuration
    // Returns: trained model pointer (caller owns, use free_model to release)
    static struct model* train(
        const std::vector<std::vector<double>>& features,
        const std::vector<int>& labels,
        const Config& config = Config()
    );

    // Train with float features (converts internally)
    static struct model* train(
        const std::vector<std::vector<float>>& features,
        const std::vector<int>& labels,
        const Config& config = Config()
    );

    // Save model to file
    static bool save_model(const struct model* m, const std::string& path);

    // Load model from file
    static struct model* load_model(const std::string& path);

    // Free model memory
    static void free_model(struct model*& m);
};

// =============================================================================
// Linear SVM Evaluator - wraps liblinear for prediction and evaluation
// =============================================================================
class LinearSVMEvaluator {
public:
    // Evaluate model on test data
    static EvalResult evaluate(
        const struct model* m,
        const std::vector<std::vector<double>>& features,
        const std::vector<int>& labels,
        int num_classes = 10
    );

    // Evaluate with float features
    static EvalResult evaluate(
        const struct model* m,
        const std::vector<std::vector<float>>& features,
        const std::vector<int>& labels,
        int num_classes = 10
    );

    // Predict single sample
    static int predict(const struct model* m, const std::vector<double>& features);
    static int predict(const struct model* m, const std::vector<float>& features);

    // Print evaluation results
    static void print_results(const EvalResult& result);
    static void print_confusion_matrix(const EvalResult& result);
};

// =============================================================================
// Utility functions for loading binary feature files
// =============================================================================
namespace utils {
    // Load features from binary file (float32, row-major)
    bool load_features_bin(
        const std::string& path,
        std::vector<std::vector<double>>& features,
        int feature_dim
    );

    // Load labels from binary file (int32)
    bool load_labels_bin(
        const std::string& path,
        std::vector<int>& labels
    );
}

} // namespace svm

#endif // LINEAR_SVM_H
