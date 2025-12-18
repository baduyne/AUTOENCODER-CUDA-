#include "dl/device.h"

namespace dl {

torch::Device get_default_device() {
#ifdef USE_CUDA
    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA);
    }
#endif
    return torch::Device(torch::kCPU);
}

bool cuda_available() {
#ifdef USE_CUDA
    return torch::cuda::is_available();
#else
    return false;
#endif
}

} // namespace dl
