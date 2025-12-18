#include "dl/svm_evaluator.h"

#include <stdexcept>
#include <iostream>
#include <unordered_map>

namespace dl {

EvaluationResult SVMEvaluator::evaluate(const svm_model*                        model,
                                        const std::vector<std::vector<double>>& features,
                                        const std::vector<int>&                 labels,
                                        int                                      num_classes)
{
    if (!model) {
        throw std::runtime_error("[SVMEvaluator] model is null");
    }
    if (features.empty() || labels.empty() || features.size() != labels.size()) {
        throw std::runtime_error("[SVMEvaluator] Invalid feature/label size.");
    }
    if (num_classes <= 0) {
        throw std::runtime_error("[SVMEvaluator] num_classes must be > 0");
    }

    EvaluationResult res;
    res.confusion_matrix.assign(num_classes, std::vector<int>(num_classes, 0));

    const int num_samples = static_cast<int>(features.size());
    int correct = 0;

    // Majority-class baseline
    std::vector<int> class_count(num_classes, 0);
    for (int y : labels) {
        if (0 <= y && y < num_classes) {
            class_count[y]++;
        }
    }
    int max_count = 0;
    for (int c = 0; c < num_classes; ++c) {
        if (class_count[c] > max_count) {
            max_count = class_count[c];
        }
    }
    res.baseline_majority_accuracy = static_cast<double>(max_count) / num_samples;

    // Predict từng sample
    for (int i = 0; i < num_samples; ++i) {
        const auto& feat = features[i];
        const int   label = labels[i];

        const int num_feat = static_cast<int>(feat.size());
        svm_node* x = new svm_node[num_feat + 1];
        for (int j = 0; j < num_feat; ++j) {
            x[j].index = j + 1;
            x[j].value = feat[j];
        }
        x[num_feat].index = -1;
        x[num_feat].value = 0.0;

        double pred = svm_predict(model, x);

        delete[] x;

        int pred_label = static_cast<int>(pred);
        if (pred_label == label) {
            ++correct;
        }

        if (0 <= label && label < num_classes &&
            0 <= pred_label && pred_label < num_classes) {
            res.confusion_matrix[label][pred_label] += 1;
        }
    }

    res.accuracy = static_cast<double>(correct) / num_samples;
    return res;
}

void SVMEvaluator::print_confusion_matrix(const EvaluationResult& res) {
    const int num_classes = static_cast<int>(res.confusion_matrix.size());
    std::cout << "Confusion Matrix (" << num_classes << "x" << num_classes << "):\n";
    std::cout << "   Pred→";
    for (int c = 0; c < num_classes; ++c) {
        std::cout << " " << c;
    }
    std::cout << "\n";

    for (int r = 0; r < num_classes; ++r) {
        std::cout << "True " << r << ":";
        for (int c = 0; c < num_classes; ++c) {
            std::cout << " " << res.confusion_matrix[r][c];
        }
        std::cout << "\n";
    }

    std::cout << "Accuracy: " << res.accuracy * 100.0 << "%\n";
    std::cout << "Baseline (majority class) accuracy: "
              << res.baseline_majority_accuracy * 100.0 << "%\n";
}

} // namespace dl
