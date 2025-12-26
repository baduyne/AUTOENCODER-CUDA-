#include "phase_entry.h"

// CPU stub wrapper when CUDA is not enabled
int main(int argc, char** argv) {
    return gpu_phase_main(argc, argv);
}
