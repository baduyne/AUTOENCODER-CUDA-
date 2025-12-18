#pragma once

#include <vector>
#include <svm.h>

namespace dl {

struct EvaluationResult {
    double accuracy = 0.0;
    std::vector<std::vector<int>> confusion_matrix; // [num_classes][num_classes]
    double baseline_majority_accuracy = 0.0;
};

class SVMEvaluator {
public:
    // Evaluate model trên test set
    // num_classes: ví dụ CIFAR-10 -> 10
    static EvaluationResult evaluate(const svm_model*                       model,
                                     const std::vector<std::vector<double>>& features,
                                     const std::vector<int>&                 labels,
                                     int                                     num_classes);

    static void print_confusion_matrix(const EvaluationResult& res);
};

} // namespace dl
