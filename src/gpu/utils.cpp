#include "utils.h"
#include <fstream>
#include <iostream>

const int IMAGE_SIZE = 32*32*3;

bool load_cifar10_images(const std::string &file_name,
                         std::vector<std::vector<float>> &images,
                         std::vector<int> &labels, int num_imgs)
{
    std::ifstream file(file_name, std::ios::binary);

    if (!file.is_open()) {
        std::cout << "Can't open file: " << file_name << "\n";
        return false;
    }

    images.resize(num_imgs, std::vector<float>(IMAGE_SIZE));
    labels.resize(num_imgs);

    for (int i = 0; i < num_imgs; i++)
    {
        uint8_t label;
        file.read((char*)&label, 1);

        labels[i] = label;

        uint8_t buffer[IMAGE_SIZE];
        file.read((char*)buffer, IMAGE_SIZE);

        for (int j = 0; j < IMAGE_SIZE; j++)
            images[i][j] = buffer[j] / 255.0f;
    }

    return true;
}
