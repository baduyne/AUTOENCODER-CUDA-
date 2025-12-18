#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>

namespace dl {

class CIFAR10Dataset {
public:
    // root: thư mục chứa data_batch_1.bin... test_batch.bin
    // train = true -> 5 file train; false -> test_batch.bin
    CIFAR10Dataset(const std::string& root, bool train);

    // tổng số sample
    std::size_t size() const;

    // reset epoch, optional shuffle index
    void reset(bool shuffle = true);

    // Lấy batch tiếp theo.
    // Trả về false nếu hết dữ liệu (hết epoch).
    bool next_batch(std::size_t batch_size,
                    const torch::Device& device,
                    torch::Tensor& images_out,
                    torch::Tensor& labels_out);

    // Truy cập tensor đầy đủ (CPU) nếu cần debug / dùng kiểu khác
    const torch::Tensor& images_cpu() const { return images_cpu_; }
    const torch::Tensor& labels_cpu() const { return labels_cpu_; }

private:
    // Tải toàn bộ dữ liệu vào CPU tensor
    void load_data(const std::string& root, bool train);

    // Tải 1 file batch CIFAR10 (ví dụ data_batch_1.bin)
    // trả về (images: [N,3,32,32], labels: [N])
    std::pair<torch::Tensor, torch::Tensor>
    load_batch_file(const std::string& path);

private:
    torch::Tensor images_cpu_;  // [N, 3, 32, 32], float32, trên CPU
    torch::Tensor labels_cpu_;  // [N], int64, trên CPU

    std::vector<int64_t> indices_; // index shuffled
    std::size_t cursor_ = 0;       // vị trí hiện tại trong indices_
};

} // namespace dl
