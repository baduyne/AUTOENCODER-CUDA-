#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>


struct Config {
    std::string weight_path = "src/gpu/weight/model_gpu.bin";
    int batch_size = 64;
    int epochs = 20;
    float lr = 0.001f;
    int patience = 2;
    std::string train_folder = "data/cifar-10-batches-bin";
    std::string test_folder  = "data/cifar-10-batches-bin";

    std::string output_folder = "./extracted_feature";

    // Optimization type: "baseline" or "loop-unroll"
    std::string optimization_type = "baseline";
    std::string log = "log";
    std::string mode = "train";
};

void parse_args(int argc, char** argv, Config& cfg);

bool load_cifar10_images(const std::string& file_name,
                         std::vector<std::vector<float>>& images,
                         std::vector<int>& labels, int num_imgs);

// Load multiple CIFAR binary files (data_batch_*.bin and/or test_batch.bin)
// Each record: 1 byte label, 3072 bytes image (3*32*32)
// The function reads all records from the provided files, normalizes pixels to [0,1],
// and appends them to `images`/`labels`.
bool load_cifar10_dataset(const std::vector<std::string>& file_list,
                          std::vector<std::vector<float>>& images,
                          std::vector<int>& labels,
                          bool shuffle = true);

// Convenience: scan a directory and load files matching data_batch*.bin and test_batch.bin
bool load_cifar10_from_dir(const std::string& dir_path,
                           std::vector<std::vector<float>>& images,
                           std::vector<int>& labels,
                           bool include_test = true,
                           bool shuffle = true);

// Shuffle images and labels in unison
void shuffle_dataset(std::vector<std::vector<float>>& images, std::vector<int>& labels);

std::vector<std::string> load_bin_files_from_folder(const std::string& folder_path);