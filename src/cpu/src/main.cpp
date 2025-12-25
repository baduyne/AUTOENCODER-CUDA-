#include <iostream>
#include <cstdlib>
#include "dl/training_pipeline_cpu.h"
#include "dl/svm_trainer.h"
#include "dl/svm_evaluator.h"

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

    // ===== Phase 2: Extract latent features =====
    // TODO: Implement extract_features trong TrainingPipelineCPU
    // (có thể chạy ở run khác, chỉ cần load lại AE)
    // pipeline.load_autoencoder(ae_save_path);
    //
    // std::vector<std::vector<double>> train_features;
    // std::vector<int>                 train_labels;
    // std::vector<std::vector<double>> test_features;
    // std::vector<int>                 test_labels;
    //
    // pipeline.extract_features(/*train=*/true,  batch_size, train_features, train_labels);
    // pipeline.extract_features(/*train=*/false, batch_size, test_features,  test_labels);
    //
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
