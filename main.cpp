#include <iostream>
#include <string>
#include "phase_entry.h"

int main(int argc, char** argv) {
    std::string phase = "cpu"; // default

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        const std::string key = "--phase";
        if (a == key && i + 1 < argc) {
            phase = argv[i + 1];
            ++i;
        } else if (a.rfind(key + "=", 0) == 0) {
            phase = a.substr(key.size() + 1);
        }
    }

    if (phase == "cpu") {
        return cpu_phase_main(argc, argv);
    } else if (phase == "gpu") {
        return gpu_phase_main(argc, argv);
    } else {
        std::cerr << "Unknown phase: " << phase << " (expected 'cpu' or 'gpu')\n";
        return 2;
    }
}
