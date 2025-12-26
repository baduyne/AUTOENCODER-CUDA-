#include "utils.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <random>
#include <algorithm>
namespace fs = std::filesystem;
const int IMAGE_SIZE = 32*32*3;


void parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        auto get_value = [&](const std::string& key) -> std::string {
            if (arg.rfind(key, 0) == 0) {
                return arg.substr(key.length());
            }
            return "";
        };

        std::string v;

        if (!(v = get_value("--weight=")).empty()) cfg.weight_path = v;
        else if (!(v = get_value("--batch=")).empty()) cfg.batch_size = std::stoi(v);
        else if (!(v = get_value("--epoch=")).empty()) cfg.epochs = std::stoi(v);
        else if (!(v = get_value("--patience=")).empty()) cfg.patience = std::stoi(v);
        else if (!(v = get_value("--lr=")).empty()) cfg.lr = std::stof(v);
        else if (!(v = get_value("--train=")).empty()) cfg.train_folder = v;
        else if (!(v = get_value("--test=")).empty()) cfg.test_folder = v;
        else if (!(v = get_value("--output=")).empty()) cfg.output_folder = v;
    }
}

std::vector<std::string> load_bin_files_from_folder(const std::string& folder_path) {
    std::vector<std::string> bin_files;

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();

            // Lọc file .bin
            if (entry.path().extension() == ".bin") {
                bin_files.push_back(path);
            }
        }
    }

    return bin_files;
}

void shuffle_dataset(std::vector<std::vector<float>>& images, std::vector<int>& labels)
{
    if (images.size() != labels.size()) return;
    std::vector<size_t> idx(images.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(idx.begin(), idx.end(), g);

    std::vector<std::vector<float>> images_shuffled;
    std::vector<int> labels_shuffled;
    images_shuffled.reserve(images.size());
    labels_shuffled.reserve(labels.size());

    for (size_t k = 0; k < idx.size(); ++k) {
        images_shuffled.push_back(std::move(images[idx[k]]));
        labels_shuffled.push_back(labels[idx[k]]);
    }

    images.swap(images_shuffled);
    labels.swap(labels_shuffled);
}


bool load_cifar10_dataset(const std::vector<std::string>& file_list,
                          std::vector<std::vector<float>>& images,
                          std::vector<int>& labels,
                          bool shuffle)
{
    images.clear();
    labels.clear();

    for (const auto &fpath : file_list) {
        std::ifstream file(fpath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Can't open file: " << fpath << "\n";
            return false;
        }

        // Read until EOF
        while (true) {
            uint8_t lbl;
            file.read((char*)&lbl, 1);
            if (!file) break; // EOF or error

            std::vector<float> img(IMAGE_SIZE);
            std::vector<uint8_t> buffer(IMAGE_SIZE);
            file.read((char*)buffer.data(), IMAGE_SIZE);
            if (!file) break; // incomplete record

            for (int j = 0; j < IMAGE_SIZE; ++j)
                img[j] = static_cast<float>(buffer[j]) / 255.0f;

            images.push_back(std::move(img));
            labels.push_back(static_cast<int>(lbl));
        }
    }

    if (shuffle) shuffle_dataset(images, labels);
    return true;
}


