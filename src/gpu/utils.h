#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>

bool load_cifar10_images(const std::string& file_name,
                         std::vector<std::vector<float>>& images,
                         std::vector<int>& labels, int num_imgs);
