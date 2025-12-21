#include "dl/cifar10_dataset.h"
#include <fstream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <iostream>

namespace dl {

CIFAR10Dataset::CIFAR10Dataset(const std::string& root, bool train, size_t max_samples) {
    max_samples_ = max_samples;
    if (!load(root, train)) {
        throw std::runtime_error("CIFAR10Dataset load failed");
    }
    if (max_samples_ > 0 && n_total_ > max_samples_) {
        // trim images_ and labels_ to max_samples_
        images_.resize(max_samples_ * 3 * 32 * 32);
        labels_.resize(max_samples_);
        n_total_ = max_samples_;
    }
    reset(true);
}

size_t CIFAR10Dataset::size() const {
    return n_total_;
}

void CIFAR10Dataset::reset(bool shuffle) {
    indices_.resize(n_total_);
    for (size_t i = 0; i < n_total_; ++i) {
        indices_[i] = i;
    }
    if (shuffle && n_total_ > 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices_.begin(), indices_.end(), gen);
    }
    cursor_ = 0;
}

bool CIFAR10Dataset::next_batch(size_t batch_size, CIFAR10Batch &batch_out) {
    if (cursor_ >= n_total_) return false;
    size_t end = std::min(cursor_ + batch_size, n_total_);
    size_t cur_batch = end - cursor_;

    // Chuẩn bị batch output
    batch_out.images = new float[cur_batch * 3 * 32 * 32];
    batch_out.labels = new uint8_t[cur_batch];
    batch_out.batch_size = cur_batch;

    for (size_t i = 0; i < cur_batch; ++i) {
        size_t idx = indices_[cursor_ + i];
        // Copy ảnh 3x32x32
        size_t img_offset = idx * 3 * 32 * 32;
        float* src_img = images_.data() + img_offset;
        float* dst_img = batch_out.images + i * 3 * 32 * 32;
        std::copy(src_img, src_img + 3 * 32 * 32, dst_img);
        // Copy label
        batch_out.labels[i] = labels_[idx];
    }
    cursor_ = end;
    return true;
}

bool CIFAR10Dataset::load(const std::string& root, bool train) {
    images_.clear(); labels_.clear();
    n_total_ = 0;
    bool ok = true;
    size_t loaded = 0;

    if (train) {
        for (int i = 1; i <= 5; ++i) {
            std::string path = root + "/data_batch_" + std::to_string(i) + ".bin";
            ok &= load_batch_file(path, images_, labels_, loaded);
            n_total_ += loaded;
        }
    } else {
        std::string path = root + "/test_batch.bin";
        ok &= load_batch_file(path, images_, labels_, loaded);
        n_total_ += loaded;
    }
    std::cout << "Loaded CIFAR10: " << n_total_ << " samples\n";
    return ok;
}

// Đọc file .bin, append vào images_buffer, labels_buffer, đặt n_loaded cho số sample đọc được
bool CIFAR10Dataset::load_batch_file(const std::string& path, std::vector<float>& images_buffer, std::vector<uint8_t>& labels_buffer, size_t& n_loaded) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open: " << path << std::endl;
        n_loaded = 0;
        return false;
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    constexpr int label_bytes = 1;
    constexpr int image_bytes = 32 * 32 * 3;
    constexpr int record_bytes = label_bytes + image_bytes;
    if (file_size % record_bytes != 0) {
        std::cerr << "Incorrect file size for: " << path << std::endl;
        n_loaded = 0;
        return false;
    }
    int n_records = static_cast<int>(file_size / record_bytes);
    std::vector<uint8_t> buffer(file_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        std::cerr << "Failed to read: " << path << std::endl;
        n_loaded = 0;
        return false;
    }
    // old_img_size was previously the number of floats in images_buffer
    // but we need the number of already-loaded samples (old_sample_count).
    size_t old_float_size = images_buffer.size();
    size_t old_lbl_size = labels_buffer.size();
    size_t old_sample_count = old_lbl_size;
    images_buffer.resize(old_float_size + n_records * 3 * 32 * 32, 0.0f);
    labels_buffer.resize(old_lbl_size + n_records, 0);
    size_t offset = 0;

    for (int i = 0; i < n_records; ++i) {
        labels_buffer[old_lbl_size + i] = buffer[offset];
        offset += label_bytes;
        // 3072 bytes: [1024R][1024G][1024B], channel-first
        int pix_per_chn = 32 * 32;
        for (int c = 0; c < 3; ++c) {
            for (int p = 0; p < pix_per_chn; ++p) {
                uint8_t pix = buffer[offset + c * pix_per_chn + p];
                float value = static_cast<float>(pix) / 255.0f;
                // compute index based on sample index (old_sample_count + i)
                size_t sample_idx = old_sample_count + i;
                int h = p / 32;
                int w = p % 32;
                size_t idx = ((sample_idx * 3 + c) * 32 + h) * 32 + w;
                images_buffer[idx] = value;
            }
        }
        offset += image_bytes;
    }
    n_loaded = n_records;
    return true;
}

} // namespace dl
