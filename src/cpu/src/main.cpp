#include <iostream>
#include <cstdlib>
#include "dl/training_pipeline_cpu.h"

int cpu_phase_main(int argc, char** argv) {
    try {
        // Cho phép override bằng environment variable hoặc dùng default
        const char* env_cifar = std::getenv("CIFAR_ROOT");
        std::string cifar_root = env_cifar ? env_cifar : "../data/cifar-10-batches-bin";
        
        std::string ae_save_path  = "./weight/model_cpu.bin";
        std::string svm_model_path= "svm_latent.model";
        
        std::cout << "Using CIFAR-10 dataset from: " << cifar_root << std::endl;

        dl::TrainingPipelineCPU pipeline(cifar_root);

    // ===== Phase 1: Train Autoencoder =====
    const char* env_epochs = std::getenv("TRAIN_EPOCHS");
    int   num_epochs    = env_epochs ? std::atoi(env_epochs) : 1; // default 1 for quick runs
    int   batch_size    = 32;
    float learning_rate = 0.01f;

    pipeline.train_autoencoder(num_epochs, batch_size, learning_rate, ae_save_path);

    // ===== Phase 2: Extract latent features and save to binary files =====
    pipeline.load_autoencoder(ae_save_path);

    // Extract and save train features (500 samples)
    pipeline.extract_and_save_features(
        /*train=*/true,
        batch_size,
        /*max_samples=*/500,
        "cpu_train_features.bin",
        "cpu_train_labels.bin"
    );

    // Extract and save test features (100 samples)
    pipeline.extract_and_save_features(
        /*train=*/false,
        batch_size,
        /*max_samples=*/100,
        "cpu_test_features.bin",
        "cpu_test_labels.bin"
    );

    std::cout << "\n[SUCCESS] CPU phase completed successfully!" << std::endl;
    std::cout << "[INFO] Features saved:" << std::endl;
    std::cout << "  - cpu_train_features.bin (500 samples)" << std::endl;
    std::cout << "  - cpu_test_features.bin (100 samples)" << std::endl;
    std::cout << "  - cpu_train_labels.bin" << std::endl;
    std::cout << "  - cpu_test_labels.bin" << std::endl;

    // // ===== Phase 3: Train SVM trên train_features =====
    // dl::SVMTrainer::Config svm_cfg;
    // svm_cfg.C      = 10.0;  // theo yêu cầu
    // svm_cfg.gamma  = -1.0;  // auto = 1/num_feat
    // svm_cfg.kernel = RBF;
    //
    // svm_model* svm = dl::SVMTrainer::train(train_features, train_labels, svm_cfg);
    // dl::SVMTrainer::save_model(svm, svm_model_path);
    // std::cout << "[INFO] SVM model saved to: " << svm_model_path << std::endl;
    //
    // // ===== Phase 4: Evaluation trên test_features =====
    // svm_model* svm_loaded = dl::SVMTrainer::load_model(svm_model_path);
    //
    // int num_classes = 10; // CIFAR-10
    // auto eval_res = dl::SVMEvaluator::evaluate(svm_loaded, test_features, test_labels, num_classes);
    // dl::SVMEvaluator::print_confusion_matrix(eval_res);
    //
        // dl::SVMTrainer::free_model(svm);
        // dl::SVMTrainer::free_model(svm_loaded);

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
