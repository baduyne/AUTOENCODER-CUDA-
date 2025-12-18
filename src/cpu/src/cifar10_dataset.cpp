#include "dl/cifar10_dataset.h"

#include <fstream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <iostream>

namespace dl {

CIFAR10Dataset::CIFAR10Dataset(const std::string& root, bool train) {
    load_data(root, train);
    reset(true);
}

std::size_t CIFAR10Dataset::size() const {
    return static_cast<std::size_t>(labels_cpu_.size(0));
}

void CIFAR10Dataset::reset(bool shuffle) {
    const auto N = size();
    indices_.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        indices_[i] = static_cast<int64_t>(i);
    }

    if (shuffle && N > 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices_.begin(), indices_.end(), gen);
    }

    cursor_ = 0;
}

bool CIFAR10Dataset::next_batch(std::size_t batch_size,
                                const torch::Device& device,
                                torch::Tensor& images_out,
                                torch::Tensor& labels_out) {
    if (cursor_ >= indices_.size()) {
        return false; // hết epoch
    }

    std::size_t end = std::min(cursor_ + batch_size, indices_.size());
    std::size_t cur_batch_size = end - cursor_;

    // Lấy indices cho batch hiện tại
    std::vector<int64_t> batch_idx(cur_batch_size);
    for (std::size_t i = 0; i < cur_batch_size; ++i) {
        batch_idx[i] = indices_[cursor_ + i];
    }
    cursor_ = end;

    torch::Tensor idx_tensor = torch::from_blob(
        batch_idx.data(),
        {static_cast<long>(cur_batch_size)},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)
    ).clone(); // clone để không phụ thuộc vào batch_idx sau khi ra khỏi scope

    // Chọn subset rồi chuyển lên device yêu cầu (CPU/GPU)
    images_out = images_cpu_.index_select(0, idx_tensor).to(device);
    labels_out = labels_cpu_.index_select(0, idx_tensor).to(device);

    return true;
}

void CIFAR10Dataset::load_data(const std::string& root, bool train) {
    std::vector<torch::Tensor> images_list;
    std::vector<torch::Tensor> labels_list;

    if (train) {
        // 5 file train: data_batch_1.bin ... data_batch_5.bin
        for (int i = 1; i <= 5; ++i) {
            std::string path = root + "/data_batch_" + std::to_string(i) + ".bin";
            auto [imgs, lbls] = load_batch_file(path);
            images_list.push_back(imgs);
            labels_list.push_back(lbls);
        }
    } else {
        std::string path = root + "/test_batch.bin";
        auto [imgs, lbls] = load_batch_file(path);
        images_list.push_back(imgs);
        labels_list.push_back(lbls);
    }

    // Gộp tất cả batch thành 1 tensor lớn
    images_cpu_ = torch::cat(images_list, /*dim=*/0); // [N_total, 3, 32, 32]
    labels_cpu_ = torch::cat(labels_list, /*dim=*/0); // [N_total]

    // đảm bảo ở CPU
    images_cpu_ = images_cpu_.to(torch::kCPU);
    labels_cpu_ = labels_cpu_.to(torch::kCPU);

    std::cout << "Loaded CIFAR10: "
              << images_cpu_.sizes() << " images, "
              << labels_cpu_.sizes() << " labels\n";
}

std::pair<torch::Tensor, torch::Tensor>
CIFAR10Dataset::load_batch_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open CIFAR10 file: " + path);
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    const int label_bytes = 1;
    const int image_bytes = 32 * 32 * 3; // 3072
    const int record_bytes = label_bytes + image_bytes;

    if (file_size % record_bytes != 0) {
        throw std::runtime_error("File size not divisible by record size for: " + path);
    }

    const int num_records = static_cast<int>(file_size / record_bytes);

    std::vector<unsigned char> buffer(file_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        throw std::runtime_error("Failed to read CIFAR10 file: " + path);
    }

    // Tạo tensor cho labels và images
    torch::Tensor labels = torch::empty({num_records}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor images = torch::empty({num_records, 3, 32, 32},
                                        torch::TensorOptions().dtype(torch::kFloat32));

    int offset = 0;
    auto labels_ptr = labels.data_ptr<int64_t>();
    auto images_ptr = images.data_ptr<float>();

    for (int i = 0; i < num_records; ++i) {
        unsigned char label = buffer[offset];
        labels_ptr[i] = static_cast<int64_t>(label);
        offset += label_bytes;

        // 3072 bytes ảnh: 1024 R, 1024 G, 1024 B
        // CIFAR-10 lưu dạng (channel-first) flattened: [R(1024), G(1024), B(1024)]
        const int pixels = 32 * 32;
        for (int c = 0; c < 3; ++c) {
            for (int p = 0; p < pixels; ++p) {
                unsigned char pix = buffer[offset + c * pixels + p];
                // normalize [0,255] -> [0,1]
                float value = static_cast<float>(pix) / 255.0f;

                // index trong tensor images: [i, c, h, w]
                int h = p / 32;
                int w = p % 32;

                std::int64_t idx =
                    ((static_cast<std::int64_t>(i) * 3 + c) * 32 + h) * 32 + w;
                images_ptr[idx] = value;
            }
        }

        offset += image_bytes;
    }

    return {images, labels};
}

} // namespace dl
