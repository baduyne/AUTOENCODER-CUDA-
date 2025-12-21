#pragma once

// Entry points for each phase. Implemented in their respective source files.
// Caller can pass through `argc, argv` if needed.

int cpu_phase_main(int argc, char** argv);
int gpu_phase_main(int argc, char** argv);
