#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdint>
#include "dl/training_pipeline_cpu.h"
#include "dl/svm_trainer.h"
#include "dl/svm_evaluator.h"

// Helper function to save features to binary file (same format as GPU)
static void save_features_bin(const std::string& path,
                              const std::vector<std::vector<double>>& features) {
    std::ofstream fout(path, std::ios::binary);
    if (!fout.is_open()) {
        std::cerr << "[ERROR] Cannot open file for writing: " << path << std::endl;
        return;
    }

    for (const auto& feat : features) {
        // Convert double to float32 for consistency with GPU output
        for (double val : feat) {
            float f = static_cast<float>(val);
            fout.write(reinterpret_cast<const char*>(&f), sizeof(float));
        }
    }
    fout.close();
    std::cout << "[INFO] Saved " << features.size() << " feature vectors to: " << path << std::endl;
}

// Helper function to save labels to binary file (same format as GPU)
static void save_labels_bin(const std::string& path,
                            const std::vector<int>& labels) {
    std::ofstream fout(path, std::ios::binary);
    if (!fout.is_open()) {
        std::cerr << "[ERROR] Cannot open file for writing: " << path << std::endl;
        return;
    }

    for (int label : labels) {
        int32_t l = static_cast<int32_t>(label);
        fout.write(reinterpret_cast<const char*>(&l), sizeof(int32_t));
    }
    fout.close();
    std::cout << "[INFO] Saved " << labels.size() << " labels to: " << path << std::endl;
}

int cpu_phase_main(int argc, char** argv) {
    try {
        // Allow override via environment variable or use default
        const char* env_cifar = std::getenv("CIFAR_ROOT");
        std::string cifar_root = env_cifar ? env_cifar : "./data/cifar-10-batches-bin";

        std::string ae_save_path = "./weight/model_cpu.bin";

        std::cout << "========================================\n"
                  << "   CPU Phase: Autoencoder Training\n"
                  << "========================================\n" << std::endl;
        std::cout << "Using CIFAR-10 dataset from: " << cifar_root << std::endl;

        dl::TrainingPipelineCPU pipeline(cifar_root);

        // ===== Phase 1: Train Autoencoder =====
        const char* env_epochs = std::getenv("TRAIN_EPOCHS");
        int   num_epochs    = env_epochs ? std::atoi(env_epochs) : 1;
        int   batch_size    = 32;
        float learning_rate = 0.01f;

        std::cout << "\n===== PHASE 1: AUTOENCODER TRAINING =====\n" << std::endl;
        pipeline.train_autoencoder(num_epochs, batch_size, learning_rate, ae_save_path);

        // ===== Phase 2: Extract latent features =====
        std::cout << "\n===== PHASE 2: FEATURE EXTRACTION =====\n" << std::endl;

        std::vector<std::vector<double>> train_features;
        std::vector<int>                 train_labels;
        std::vector<std::vector<double>> test_features;
        std::vector<int>                 test_labels;

        std::cout << "[INFO] Extracting training features..." << std::endl;
        pipeline.extract_features(/*train=*/true, batch_size, train_features, train_labels);
        std::cout << "[INFO] Extracted " << train_features.size() << " training samples" << std::endl;

        std::cout << "[INFO] Extracting test features..." << std::endl;
        pipeline.extract_features(/*train=*/false, batch_size, test_features, test_labels);
        std::cout << "[INFO] Extracted " << test_features.size() << " test samples" << std::endl;

        // ===== Phase 3: Save features to binary files (same format as GPU) =====
        std::cout << "\n===== PHASE 3: SAVING FEATURES TO BINARY FILES =====\n" << std::endl;

        save_features_bin("train_features.bin", train_features);
        save_labels_bin("train_labels.bin", train_labels);
        save_features_bin("test_features.bin", test_features);
        save_labels_bin("test_labels.bin", test_labels);

        std::cout << "\n========================================\n"
                  << "   CPU Phase Complete!\n"
                  << "========================================\n"
                  << "\nTo run SVM training/evaluation, use:\n"
                  << "  ./svm_main\n"
                  << "\nOr with custom parameters:\n"
                  << "  ./svm_main --C 10.0 --gamma auto --model svm_model.bin\n"
                  << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[FATAL ERROR] Unknown exception occurred" << std::endl;
        return 1;
    }
}

// Note: no standalone `main` here; use root-level launcher to call `cpu_phase_main`.
