#include <iostream>

// Stub implementation khi USE_CUDA=OFF
int gpu_phase_main(int argc, char** argv) {
    std::cerr << "ERROR: GPU phase requires CUDA support. Please build with USE_CUDA=ON" << std::endl;
    std::cerr << "Example: cmake -DUSE_CUDA=ON .." << std::endl;
    return 1;
}

