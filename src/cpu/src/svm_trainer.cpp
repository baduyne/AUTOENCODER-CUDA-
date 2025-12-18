#include "dl/svm_trainer.h"

#include <stdexcept>
#include <iostream>

namespace dl {

svm_model* SVMTrainer::train(const std::vector<std::vector<double>>& features,
                             const std::vector<int>&                 labels,
                             const Config&                           config)
{
    if (features.empty() || labels.empty() || features.size() != labels.size()) {
        throw std::runtime_error("[SVMTrainer] Invalid feature/label size.");
    }

    const int num_samples = static_cast<int>(features.size());
    const int num_feat    = static_cast<int>(features[0].size());

    // ----- Build svm_problem -----
    svm_problem prob{};
    prob.l = num_samples;

    prob.y = new double[prob.l];
    prob.x = new svm_node*[prob.l];

    for (int i = 0; i < num_samples; ++i) {
        prob.y[i] = static_cast<double>(labels[i]);

        svm_node* x_space = new svm_node[num_feat + 1];
        for (int j = 0; j < num_feat; ++j) {
            x_space[j].index = j + 1;         // LIBSVM index bắt đầu từ 1
            x_space[j].value = features[i][j];
        }
        x_space[num_feat].index = -1;         // sentinel
        x_space[num_feat].value = 0.0;

        prob.x[i] = x_space;
    }

    // ----- SVM parameters -----
    svm_parameter param{};
    param.svm_type    = C_SVC;
    param.kernel_type = config.kernel;        // RBF
    param.degree      = 3;
    param.gamma       = (config.gamma > 0.0)
                        ? config.gamma
                        : 1.0 / static_cast<double>(num_feat); // gamma=auto
    param.coef0       = 0;
    param.nu          = 0.5;
    param.cache_size  = 200;   // MB
    param.C           = config.C;   // C = 10
    param.eps         = 1e-3;
    param.p           = 0.1;
    param.shrinking   = 1;
    param.probability = 0;
    param.nr_weight   = 0;
    param.weight_label = nullptr;
    param.weight       = nullptr;

    // Check parameter
    const char* err_msg = svm_check_parameter(&prob, &param);
    if (err_msg) {
        for (int i = 0; i < num_samples; ++i) {
            delete[] prob.x[i];
        }
        delete[] prob.x;
        delete[] prob.y;

        throw std::runtime_error(std::string("[SVMTrainer] Parameter error: ") + err_msg);
    }

    // ----- Train -----
    std::cout << "===== TRAIN LIBSVM (C=" << param.C
              << ", gamma=" << param.gamma << ") =====" << std::endl;

    svm_model* model = svm_train(&prob, &param);
    if (!model) {
        for (int i = 0; i < num_samples; ++i) {
            delete[] prob.x[i];
        }
        delete[] prob.x;
        delete[] prob.y;

        throw std::runtime_error("[SVMTrainer] Training failed (model is null).");
    }

    // Free problem memory (model không dùng lại prob.x/prob.y)
    for (int i = 0; i < num_samples; ++i) {
        delete[] prob.x[i];
    }
    delete[] prob.x;
    delete[] prob.y;

    return model;
}

// Overload: dùng config mặc định
svm_model* SVMTrainer::train(const std::vector<std::vector<double>>& features,
                             const std::vector<int>&                 labels)
{
    Config default_config;
    return train(features, labels, default_config);
}

void SVMTrainer::save_model(const svm_model* model, const std::string& path) {
    if (!model) {
        throw std::runtime_error("[SVMTrainer] save_model: model is null");
    }
    if (svm_save_model(path.c_str(), model) != 0) {
        throw std::runtime_error("[SVMTrainer] Failed to save model to: " + path);
    }
}

svm_model* SVMTrainer::load_model(const std::string& path) {
    svm_model* model = svm_load_model(path.c_str());
    if (!model) {
        throw std::runtime_error("[SVMTrainer] Failed to load model from: " + path);
    }
    return model;
}

void SVMTrainer::free_model(svm_model*& model) {
    if (model) {
        svm_free_and_destroy_model(&model);
        model = nullptr;
    }
}

} // namespace dl
