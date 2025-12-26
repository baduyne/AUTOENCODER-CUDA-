/**
 * Unified SVM Phase for CPU and GPU Feature Comparison
 *
 * Uses liblinear for fast linear SVM classification
 * Supports loading features from binary files produced by both CPU and GPU pipelines
 */

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

#include "linear_svm.h"

// Feature dimension from autoencoder latent space: 128 * 8 * 8
static const int FEATURE_DIM = 8192;

// =============================================================================
// Configuration
// =============================================================================

struct AppConfig {
    // Feature file paths
    std::string train_features_path = "train_features.bin";
    std::string train_labels_path   = "train_labels.bin";
    std::string test_features_path  = "test_features.bin";
    std::string test_labels_path    = "test_labels.bin";
    std::string model_path          = "linear_svm_model.bin";

    // SVM parameters
    double C = 0.1;
    double eps = 0.01;
    int solver_type = L2R_L2LOSS_SVC_DUAL;  // Default: L2-loss SVC dual
    int num_classes = 10;

    // Mode flags
    bool train_only = false;
    bool eval_only = false;
    bool compare_mode = false;  // Compare CPU vs GPU features

    // For comparison mode
    std::string cpu_features_prefix = "cpu_";
    std::string gpu_features_prefix = "";  // GPU is default (no prefix)
};

// =============================================================================
// CLI Parser
// =============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "Unified SVM Phase - Linear SVM for CPU/GPU feature classification\n"
              << "Uses liblinear for fast training and evaluation.\n"
              << "\nFile Options:\n"
              << "  --train-features PATH   Training features (default: train_features.bin)\n"
              << "  --train-labels PATH     Training labels (default: train_labels.bin)\n"
              << "  --test-features PATH    Test features (default: test_features.bin)\n"
              << "  --test-labels PATH      Test labels (default: test_labels.bin)\n"
              << "  --model PATH            Model save/load path (default: linear_svm_model.bin)\n"
              << "\nSVM Parameters:\n"
              << "  --C VALUE               Regularization parameter (default: 1.0)\n"
              << "  --eps VALUE             Stopping tolerance (default: 0.1)\n"
              << "  --solver TYPE           Solver type 0-7 (default: 1)\n"
              << "                          0: L2R_LR (Logistic Regression)\n"
              << "                          1: L2R_L2LOSS_SVC_DUAL (L2-loss SVC dual)\n"
              << "                          2: L2R_L2LOSS_SVC (L2-loss SVC primal)\n"
              << "                          3: L2R_L1LOSS_SVC_DUAL (L1-loss SVC dual)\n"
              << "                          4: MCSVM_CS (Crammer-Singer)\n"
              << "                          5: L1R_L2LOSS_SVC (L1-regularized L2-loss)\n"
              << "                          6: L1R_LR (L1-regularized LR)\n"
              << "                          7: L2R_LR_DUAL (LR dual)\n"
              << "  --num-classes N         Number of classes (default: 10)\n"
              << "\nMode Options:\n"
              << "  --train-only            Train only, skip evaluation\n"
              << "  --eval-only             Evaluate only using existing model\n"
              << "  --compare               Compare CPU vs GPU features\n"
              << "                          Expects: cpu_train_features.bin, train_features.bin, etc.\n"
              << "\n  --help                  Show this help message\n"
              << std::endl;
}

bool parse_args(int argc, char** argv, AppConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        }
        else if (arg == "--train-features" && i + 1 < argc) {
            config.train_features_path = argv[++i];
        }
        else if (arg == "--train-labels" && i + 1 < argc) {
            config.train_labels_path = argv[++i];
        }
        else if (arg == "--test-features" && i + 1 < argc) {
            config.test_features_path = argv[++i];
        }
        else if (arg == "--test-labels" && i + 1 < argc) {
            config.test_labels_path = argv[++i];
        }
        else if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        }
        else if (arg == "--C" && i + 1 < argc) {
            config.C = std::stod(argv[++i]);
        }
        else if (arg == "--eps" && i + 1 < argc) {
            config.eps = std::stod(argv[++i]);
        }
        else if (arg == "--solver" && i + 1 < argc) {
            config.solver_type = std::stoi(argv[++i]);
        }
        else if (arg == "--num-classes" && i + 1 < argc) {
            config.num_classes = std::stoi(argv[++i]);
        }
        else if (arg == "--train-only") {
            config.train_only = true;
        }
        else if (arg == "--eval-only") {
            config.eval_only = true;
        }
        else if (arg == "--compare") {
            config.compare_mode = true;
        }
        else {
            std::cerr << "[Warning] Unknown argument: " << arg << std::endl;
        }
    }
    return true;
}

// =============================================================================
// Single Pipeline Run
// =============================================================================

bool run_svm_pipeline(
    const AppConfig& config,
    const std::string& name = "")
{
    using Clock = std::chrono::high_resolution_clock;

    std::string prefix = name.empty() ? "" : "[" + name + "] ";

    std::cout << "\n========================================\n"
              << prefix << "Linear SVM Pipeline\n"
              << "========================================\n" << std::endl;

    std::cout << prefix << "Config:\n"
              << "  Train features: " << config.train_features_path << "\n"
              << "  Test features:  " << config.test_features_path << "\n"
              << "  Model path:     " << config.model_path << "\n"
              << "  C=" << config.C << ", eps=" << config.eps
              << ", solver=" << config.solver_type << "\n" << std::endl;

    struct model* svm_model = nullptr;
    svm::NormalizationStats norm_stats;  // Store normalization stats for test set

    try {
        // =================================================================
        // Training Phase
        // =================================================================
        if (!config.eval_only) {
            std::cout << prefix << "===== TRAINING PHASE =====\n" << std::endl;

            // Load training data
            std::vector<std::vector<double>> train_features;
            std::vector<int> train_labels;

            if (!svm::utils::load_features_bin(config.train_features_path, train_features, FEATURE_DIM)) {
                return false;
            }
            if (!svm::utils::load_labels_bin(config.train_labels_path, train_labels)) {
                return false;
            }

            if (train_features.size() != train_labels.size()) {
                std::cerr << prefix << "Error: Feature/label size mismatch" << std::endl;
                return false;
            }

            // IMPORTANT: Normalize features (Z-score normalization)
            std::cout << prefix << "Normalizing training features..." << std::endl;
            norm_stats = svm::utils::normalize_features_inplace(train_features);

            // Configure SVM
            svm::Config svm_config;
            svm_config.C = config.C;
            svm_config.eps = config.eps;
            svm_config.solver_type = config.solver_type;

            // Train
            auto train_start = Clock::now();
            svm_model = svm::LinearSVMTrainer::train(train_features, train_labels, svm_config);
            auto train_end = Clock::now();

            double train_time = std::chrono::duration<double>(train_end - train_start).count();
            std::cout << prefix << "Training time: " << std::fixed << std::setprecision(2)
                      << train_time << " seconds" << std::endl;

            if (!svm_model) {
                std::cerr << prefix << "Training failed!" << std::endl;
                return false;
            }

            // Save model
            svm::LinearSVMTrainer::save_model(svm_model, config.model_path);
        }

        // =================================================================
        // Evaluation Phase
        // =================================================================
        if (!config.train_only) {
            std::cout << "\n" << prefix << "===== EVALUATION PHASE =====\n" << std::endl;

            // Load model if eval-only mode
            if (config.eval_only) {
                svm_model = svm::LinearSVMTrainer::load_model(config.model_path);
                if (!svm_model) {
                    return false;
                }
            }

            // Load test data
            std::vector<std::vector<double>> test_features;
            std::vector<int> test_labels;

            if (!svm::utils::load_features_bin(config.test_features_path, test_features, FEATURE_DIM)) {
                return false;
            }
            if (!svm::utils::load_labels_bin(config.test_labels_path, test_labels)) {
                return false;
            }

            // IMPORTANT: Normalize test features using training set statistics
            std::cout << prefix << "Normalizing test features using training statistics..." << std::endl;
            svm::utils::normalize_features(test_features, norm_stats);

            // Evaluate
            auto eval_start = Clock::now();
            auto result = svm::LinearSVMEvaluator::evaluate(
                svm_model, test_features, test_labels, config.num_classes);
            auto eval_end = Clock::now();

            double eval_time = std::chrono::duration<double>(eval_end - eval_start).count();
            std::cout << prefix << "Evaluation time: " << std::fixed << std::setprecision(2)
                      << eval_time << " seconds" << std::endl;

            // Print results
            svm::LinearSVMEvaluator::print_results(result);
            svm::LinearSVMEvaluator::print_confusion_matrix(result);
        }

        // Cleanup
        if (svm_model) {
            svm::LinearSVMTrainer::free_model(svm_model);
        }

    } catch (const std::exception& e) {
        std::cerr << prefix << "Error: " << e.what() << std::endl;
        if (svm_model) {
            svm::LinearSVMTrainer::free_model(svm_model);
        }
        return false;
    }

    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================\n"
              << "   Linear SVM Phase (liblinear)\n"
              << "   For CPU/GPU Feature Classification\n"
              << "========================================\n" << std::endl;

    AppConfig config;
    if (!parse_args(argc, argv, config)) {
        return 0;  // --help was called
    }

    // Compare mode: run both CPU and GPU features
    if (config.compare_mode) {
        std::cout << "Running in COMPARE MODE\n"
                  << "Will evaluate both CPU and GPU features\n" << std::endl;

        // GPU features (default names)
        AppConfig gpu_config = config;
        gpu_config.model_path = "gpu_" + config.model_path;

        // CPU features (with prefix)
        AppConfig cpu_config = config;
        cpu_config.train_features_path = "cpu_" + config.train_features_path;
        cpu_config.train_labels_path = "cpu_" + config.train_labels_path;
        cpu_config.test_features_path = "cpu_" + config.test_features_path;
        cpu_config.test_labels_path = "cpu_" + config.test_labels_path;
        cpu_config.model_path = "cpu_" + config.model_path;

        std::cout << "========================================\n"
                  << "          GPU FEATURES\n"
                  << "========================================" << std::endl;
        bool gpu_ok = run_svm_pipeline(gpu_config, "GPU");

        std::cout << "\n========================================\n"
                  << "          CPU FEATURES\n"
                  << "========================================" << std::endl;
        bool cpu_ok = run_svm_pipeline(cpu_config, "CPU");

        if (gpu_ok && cpu_ok) {
            std::cout << "\n========================================\n"
                      << "   COMPARISON COMPLETE\n"
                      << "========================================\n" << std::endl;
        }

        return (gpu_ok && cpu_ok) ? 0 : 1;
    }

    // Single run mode
    bool success = run_svm_pipeline(config);

    if (success) {
        std::cout << "[INFO] SVM phase completed successfully!" << std::endl;
    }

    return success ? 0 : 1;
}
