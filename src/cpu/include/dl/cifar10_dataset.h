#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace dl {

// Batch structure cho 1 lần lấy batch
struct CIFAR10Batch {
    float* images = nullptr;       // Pointer tới ảnh, shape: [batch_size, 3, 32, 32] (row-major, C contiguous)
    uint8_t* labels = nullptr;     // Pointer tới nhãn, shape: [batch_size]
    size_t batch_size = 0;   // Kích thước thực tế của batch (có thể < batch_size cuối cùng của epoch)
    
    ~CIFAR10Batch() {
        if (images) delete[] images;
        if (labels) delete[] labels;
    }
};

class CIFAR10Dataset {
public:
    // root: thư mục chứa data_batch_1.bin... test_batch.bin
    // train = true -> 5 file train; false -> test_batch.bin
    CIFAR10Dataset(const std::string& root, bool train, size_t max_samples = 0);

    size_t size() const; // tổng số sample
    void reset(bool shuffle = true); // reset epoch, optional shuffle index
    bool next_batch(size_t batch_size, CIFAR10Batch& batch_out); // trả về false nếu hết epoch

    // Truy cập toàn bộ buffer ảnh/nhãn (không shuffle, dùng để debug)
    const std::vector<float>& images() const { return images_; }
    const std::vector<uint8_t>& labels() const { return labels_; }

private:
    bool load(const std::string& root, bool train);            // load toàn bộ dữ liệu
    bool load_batch_file(const std::string& path,              // đọc 1 file binary
                         std::vector<float>& images_buffer,
                         std::vector<uint8_t>& labels_buffer,
                         size_t& n_loaded);

private:
    std::vector<float> images_;        // [N,3,32,32] C order
    std::vector<uint8_t> labels_;      // [N]
    std::vector<size_t> indices_;      // shuffle index
    size_t cursor_ = 0;                // vị trí hiện tại trong indices_
    size_t n_total_ = 0;               // số sample tổng
    size_t max_samples_ = 0;           // optional cap on samples to load (0 = no cap)
};

} // namespace dl
