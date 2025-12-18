#pragma once

#include <vector>
#include <string>
#include <svm.h>

namespace dl {

class SVMTrainer {
public:
    struct Config {
        double C      = 10.0;  // theo yêu cầu
        double gamma  = -1.0;  // <0 => auto = 1/num_feat
        int    kernel = RBF;   // RBF kernel
    };

    // Train SVM từ feature + label
    // features: [N][F], labels: [N]
    // Trả về svm_model* (caller chịu trách nhiệm free)
    
    // Overload 1: Với custom config
    static svm_model* train(const std::vector<std::vector<double>>& features,
                            const std::vector<int>&                 labels,
                            const Config&                           config);
    
    // Overload 2: Dùng config mặc định
    static svm_model* train(const std::vector<std::vector<double>>& features,
                            const std::vector<int>&                 labels);

    // Save & load model
    static void save_model(const svm_model* model, const std::string& path);
    static svm_model* load_model(const std::string& path);

    // Free model
    static void free_model(svm_model*& model);
};

} // namespace dl
