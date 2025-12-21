#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>

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
